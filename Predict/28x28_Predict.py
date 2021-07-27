import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import Models.HiCSR as hicsr
from math import log10
from math import exp
import torch
import torch.nn.functional as F
import gzip
import copy
from sklearn import metrics
import scipy.sparse as sps

from Utils.io import spreadM, together

from Data.all_parser import *


# Adjust 40x40 data for HiCSR 28x28 output
def predict(model, data, input_size=40, output_size=28):
    left_pad_value = int((input_size - output_size) / 2)
    right_pad_value = left_pad_value + (output_size - (data.shape[2] % output_size))
    padded_data = F.pad(data, (6, 6, 6, 6), mode='constant')
    predicted_mat = torch.zeros((1, 1, padded_data.shape[2], padded_data.shape[3]))
    predicted_mat = model(padded_data).to(device)
    return predicted_mat


def to_transition(mtogether):
    sums = mtogether.sum(axis=1)
    # make the ones that are 0, so that we don't divide by 0
    sums[sums == 0.0] = 1.0
    D = sps.spdiags(1.0 / sums.flatten(), [0], mtogether.shape[0], mtogether.shape[1], format='csr')
    return D.dot(mtogether)


def random_walk(m_input, t):
    # return m_input.__pow__(t)
    # return np.linalg.matrix_power(m_input,t)
    return m_input.__pow__(t)


def write_diff_vector_bedfile(diff_vector, nodes, nodes_idx, out_filename):
    out = gzip.open(out_filename, 'w')
    for i in range(diff_vector.shape[0]):
        node_name = nodes_idx[i]
        node_dict = nodes[node_name]
        out.write(str(node_dict['chr']) + '\t' + str(node_dict['start']) + '\t' + str(
            node_dict['end']) + '\t' + node_name + '\t' + str(diff_vector[i][0]) + '\n')
    out.close()


def compute_reproducibility(m1_csr, m2_csr, transition, tmax=3, tmin=3):
    # make symmetric
    m1up = m1_csr
    m1down = m1up.transpose()
    m1 = m1up + m1down

    m2up = m2_csr
    m2down = m2up.transpose()
    m2 = m2up + m2down

    # convert to an actual transition matrix
    if transition:
        m1 = to_transition(m1)
        m2 = to_transition(m2)

    # count nonzero nodes (note that we take the average number of nonzero nodes in the 2 datasets)
    rowsums_1 = m1.sum(axis=1)
    nonzero_1 = [i for i in range(rowsums_1.shape[0]) if rowsums_1[i] > 0.0]
    rowsums_2 = m2.sum(axis=1)
    nonzero_2 = [i for i in range(rowsums_2.shape[0]) if rowsums_2[i] > 0.0]
    nonzero_total = len(list(set(nonzero_1).union(set(nonzero_2))))
    nonzero_total = 0.5 * (1.0 * len(list(set(nonzero_1))) + 1.0 * len(list(set(nonzero_2))))

    scores = []
    if True:
        diff_vector = np.zeros((m1.shape[0], 1))
        for t in range(1, tmax + 1):  # range(args.tmin,args.tmax+1):
            extra_text = ' (not included in score calculation)'
            if t == 1:
                rw1 = copy.deepcopy(m1)
                rw2 = copy.deepcopy(m2)

            else:
                rw1 = rw1.dot(m1)
                rw2 = rw2.dot(m2)

            if t >= tmin:
                # diff_vector += (abs(rw1 - rw2)).sum(axis=1)
                diff = abs(rw1 - rw2).sum()  # +euclidean(rw1.toarray().flatten(),rw2.toarray().flatten()))
                scores.append(1.0 * float(diff) / float(nonzero_total))
                extra_text = ' | score=' + str('{:.3f}'.format(1.0 - float(diff) / float(nonzero_total)))
    #             print('GenomeDISCO | ' + strftime("%c") + ' | done t=' + str(t) + extra_text)

    # compute final score
    ts = range(tmin, tmax + 1)
    denom = len(ts) - 1
    if tmin == tmax:
        auc = scores[0]

        if 2 < auc:
            auc = 2

        elif 0 <= auc <= 2:
            auc = auc

    else:
        auc = metrics.auc(range(len(ts)), scores) / denom

    reproducibility = 1 - auc
    return reproducibility


############### SSIM ###############
def _toimg(mat):
    m = torch.tensor(mat)
    # convert to float and add channel dimension
    return m.float()


def _tohic(mat):
    mat.squeeze_()
    return mat.numpy()  # .astype(int)


def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x - width // 2) ** 2 / float(2 * sigma ** 2)) for x in range(width)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def gaussian_filter(img, width, sigma=3):
    img = _toimg(img).unsqueeze(0)
    _, channel, _, _ = img.size()
    window = create_window(width, channel, sigma)
    mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
    return _tohic(mu1)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size()[1]
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


############### Predict ###############

def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def hicarn_predictor(deephic_loader, ckpt_file, device):
    deepmodel = hicsr.Generator().to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file))
    print(f'Loading HiCSR checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []
    test_metrics = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    ssims = []
    psnrs = []
    mses = []
    repro = []

    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(deephic_loader, desc='HiCSR Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            test_metrics['nsamples'] += batch_size
            lr = lr.to(device)
            hr = hr.to(device)
            out = predict(deepmodel, lr)

            batch_mse = ((out - hr) ** 2).mean()
            test_metrics['mse'] += batch_mse * batch_size
            batch_ssim = ssim(out, hr)
            test_metrics['ssims'] += batch_ssim * batch_size
            test_metrics['psnr'] = 10 * log10(1 / (test_metrics['mse'] / test_metrics['nsamples']))
            test_metrics['ssim'] = test_metrics['ssims'] / test_metrics['nsamples']
            tqdm(deephic_loader, desc='HiCNN Predicting: ').set_description(
                desc=f"[Predicting in Test set] PSNR: {test_metrics['psnr']:.4f} dB SSIM: {test_metrics['ssim']:.4f}")

            for i, j in zip(hr, out):
                out1 = torch.squeeze(j, dim=0)
                hr1 = torch.squeeze(i, dim=0)
                out2 = out1.cpu().detach().numpy()
                hr2 = hr1.cpu().detach().numpy()
                genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
                repro.append(genomeDISCO)

            ssims.append(test_metrics['ssim'])
            psnrs.append(test_metrics['psnr'])
            mses.append(batch_mse)

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    mean_ssim = sum(ssims) / len(ssims)
    mean_psnr = sum(psnrs) / len(psnrs)
    mean_mse = sum(mses) / len(mses)
    mean_repro = sum(repro) / len(repro)

    print("Mean SSIM: ", mean_ssim)
    print("Mean PSNR: ", mean_psnr)
    print("Mean MSE: ", mean_mse)
    print("GenomeDISCO Score: ", mean_repro)
    deep_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return deep_hics


def save_data(deep_hic, compact, size, file):
    deephic = spreadM(deep_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, deephic=deephic, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    res_num = args.resblock
    cuda = args.cuda
    print('WARNING: Prediction process requires a large memory. Ensure that your machine has ~150G of memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    in_dir = os.path.join(root_dir, 'data')
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]
    deephic_file = os.path.join(root_dir, 'data/', args.file)

    chunk, stride, bound, scale = filename_parser(deephic_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data[DeepHiC]: {deephic_file}')
    deephic_data = np.load(os.path.join(in_dir, deephic_file), allow_pickle=True)
    deephic_loader = dataloader(deephic_data)

    indices, compacts, sizes = data_info(deephic_data)
    deep_hics = hicarn_predictor(deephic_loader, ckpt_file, device)


    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(deep_hics[key], compacts[key], sizes[key], file)


    pool = multiprocessing.Pool(processes=3)
    print(f'Start a multiprocess pool with process_num = 3 for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
