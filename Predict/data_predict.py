import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
from math import exp
import torch
import torch.nn.functional as F

from Data.all_parser import *


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


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    print(inputs.size())
    print(target.size())
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def divide(mat, chr_num, chunk_size=40, stride=28, bound=201, padding=True, verbose=False):
    chr_str = str(chr_num)
    result = []
    index = []
    size = mat.shape[0]
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len,pad_len), (pad_len,pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j)<=bound and i+chunk_size<height and j+chunk_size<width:
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(f'[Chr{chr_str}] Deviding HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


# def dataloader_deep(data1, data2, batch_size=64):
#     # data1 = np.array(data1['deephic'])
#     # data2 = np.array(data2['hic'])
#     # data1 = divide(data1, chr_num=14)
#     # data2 = divide(data2, chr_num=14)
#     inds = torch.tensor(data1[1], dtype=torch.long)
#     prediction = torch.tensor(data1[0], dtype=torch.float)
#     target = torch.tensor(data2[0], dtype=torch.float)
#     print(prediction.size())
#     print(target.size())
#     dataset = TensorDataset(prediction, target, inds)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     return loader


def deephic_analysis(deephic_loader, device):
    result_data = []
    result_inds = []
    test_metrics = {'g_loss': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    ssims = []
    psnrs = []
    mses = []
    with torch.no_grad():
        for batch in tqdm(deephic_loader, desc='DeepHiC Predicting: '):
            sr, hr, inds = batch
            batch_size = sr.size(0)
            test_metrics['nsamples'] += batch_size
            sr = sr.to(device)
            hr = hr.to(device)

            batch_mse = ((sr - hr) ** 2).mean()
            test_metrics['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            test_metrics['ssims'] += batch_ssim * batch_size
            test_metrics['psnr'] = 10 * log10(1 / (test_metrics['mse'] / test_metrics['nsamples']))
            test_metrics['ssim'] = test_metrics['ssims'] / test_metrics['nsamples']
            tqdm(deephic_loader, desc='DeepHiC Predicting: ').set_description(
                desc=f"[Predicting in Test set] PSNR: {test_metrics['psnr']:.4f} dB SSIM: {test_metrics['ssim']:.4f}")

            ssims.append(test_metrics['ssim'])
            psnrs.append(test_metrics['psnr'])
            mses.append(batch_mse)

            result_data.append(sr.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    mean_ssim = sum(ssims) / len(ssims)
    mean_psnr = sum(psnrs) / len(psnrs)
    mean_mse = sum(mses) / len(mses)

    print("Mean SSIM: ", mean_ssim)
    print("Mean PSNR: ", mean_psnr)
    print("Mean MSE: ", mean_mse)
    return mean_ssim, mean_psnr, mean_mse

args = data_predict_parser().parse_args(sys.argv[1:])
cuda = args.cuda
device = torch.device(
    f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
print(f'Using device: {device}')

# data1 = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/DeepHiC_Predict/predict_chr14_40kb.npz')
# data2 = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr14_10kb.npz')
data = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/data/deephic_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test_CARN_Chr14_40.npz')

compare_data = dataloader(data, batch_size=64)
deephic_analysis(compare_data, device=device)

# def data_info(data):
#     indices = data['inds']
#     compacts = data['compacts'][()]
#     sizes = data['sizes'][()]
#     return indices, compacts, sizes
#
#
# get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))
#
#
# def filename_parser(filename):
#     info_str = filename.split('.')[0].split('_')[2:-1]
#     chunk = get_digit(info_str[0])
#     stride = get_digit(info_str[1])
#     bound = get_digit(info_str[2])
#     scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
#     return chunk, stride, bound, scale
#
#
# # Add resblock_num=res_num when running DeepHiC Generator
# def deephic_predictor(deephic_loader, ckpt_file, scale, res_num, device):
#     deepmodel = deephic.Generator(scale_factor=1).to(device)
#     if not os.path.isfile(ckpt_file):
#         ckpt_file = f'save/{ckpt_file}'
#     deepmodel.load_state_dict(torch.load(ckpt_file))
#     print(f'Loading DeepHiC checkpoint file from "{ckpt_file}"')
#     result_data = []
#     result_inds = []
#     test_metrics = {'g_loss': 0,
#                     'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
#     ssims = []
#     psnrs = []
#     mses = []
#     deepmodel.eval()
#     with torch.no_grad():
#         for batch in tqdm(deephic_loader, desc='DeepHiC Predicting: '):
#             lr, hr, inds = batch
#             batch_size = lr.size(0)
#             test_metrics['nsamples'] += batch_size
#             lr = lr.to(device)
#             hr = hr.to(device)
#             out = deepmodel(lr)
#
#
#             batch_mse = ((out - hr) ** 2).mean()
#             test_metrics['mse'] += batch_mse * batch_size
#             batch_ssim = ssim(out, hr)
#             test_metrics['ssims'] += batch_ssim * batch_size
#             test_metrics['psnr'] = 10 * log10(1 / (test_metrics['mse'] / test_metrics['nsamples']))
#             test_metrics['ssim'] = test_metrics['ssims'] / test_metrics['nsamples']
#             tqdm(deephic_loader, desc='DeepHiC Predicting: ').set_description(
#                 desc=f"[Predicting in Test set] PSNR: {test_metrics['psnr']:.4f} dB SSIM: {test_metrics['ssim']:.4f}")
#
#             ssims.append(test_metrics['ssim'])
#             psnrs.append(test_metrics['psnr'])
#             mses.append(batch_mse)
#
#             result_data.append(out.to('cpu').numpy())
#             result_inds.append(inds.numpy())
#     result_data = np.concatenate(result_data, axis=0)
#     result_inds = np.concatenate(result_inds, axis=0)
#     mean_ssim = sum(ssims) / len(ssims)
#     mean_psnr = sum(psnrs) / len(psnrs)
#     mean_mse = sum(mses) / len(mses)
#
#     print("Mean SSIM: ", mean_ssim)
#     print("Mean PSNR: ", mean_psnr)
#     print("Mean MSE: ", mean_mse)
#     deep_hics = together(result_data, result_inds, tag='Reconstructing: ')
#     return deep_hics
#
#
# def save_data(deep_hic, compact, size, file):
#     deephic = spreadM(deep_hic, compact, size, convert_int=False, verbose=True)
#     np.savez_compressed(file, deephic=deephic, compact=compact)
#     print('Saving file:', file)
#
#
# if __name__ == '__main__':
#     args = data_predict_parser().parse_args(sys.argv[1:])
#     cell_line = args.cell_line
#     low_res = args.low_res
#     ckpt_file = args.checkpoint
#     res_num = args.resblock
#     cuda = args.cuda
#     print('WARNING: Predict process needs large memory, thus ensure that your machine have ~150G memory.')
#     if multiprocessing.cpu_count() > 23:
#         pool_num = 23
#     else:
#         exit()
#
#     in_dir = os.path.join(root_dir, 'data')
#     out_dir = os.path.join(root_dir, 'predict', cell_line)
#     mkdir(out_dir)
#
#     files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]
#     # deephic_file = [f for f in files if f.find(cell_line.lower() + '.npz') >= 0][0]
#     deephic_file = 'deephic_10kb40kb_c40_s40_b201_nonpool_human_GM12878_Chr16_40.npz'
#
#     chunk, stride, bound, scale = filename_parser(deephic_file)
#
#     device = torch.device(
#         f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
#     print(f'Using device: {device}')
#
#     start = time.time()
#     print(f'Loading data[DeepHiC]: {deephic_file}')
#     deephic_data = np.load(os.path.join(in_dir, deephic_file), allow_pickle=True)
#     deephic_loader = dataloader(deephic_data)
#
#     indices, compacts, sizes = data_info(deephic_data)
#     deep_hics = deephic_predictor(deephic_loader, ckpt_file, scale, res_num, device)
#
#
#     def save_data_n(key):
#         file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
#         save_data(deep_hics[key], compacts[key], sizes[key], file)
#
#
#     pool = multiprocessing.Pool(processes=pool_num)
#     print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
#     for key in compacts.keys():
#         pool.apply_async(save_data_n, (key,))
#     pool.close()
#     pool.join()
#     print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
