import sys
import time
import multiprocessing
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from Utils.ssim import ssim
from Models import CARN as CARN
from math import log10

from Utils.io import spreadM, together

from Data.all_parser import *


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


# Add resblock_num=res_num when running DeepHiC Generator
def deephic_predictor(carn_loader, ckpt_file, scale, res_num, device):
    deepmodel = CARN.Generator(scale_factor=scale, num_channels=64).to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file))
    print(f'Loading CARN checkpoint file from "{ckpt_file}"')
    result_data = []
    result_inds = []
    test_metrics = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(carn_loader, desc='CARN Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            test_metrics['nsamples'] += batch_size
            lr = lr.to(device)
            hr = hr.to(device)
            out = deepmodel(lr)

            batch_mse = ((out - hr) ** 2).mean()
            test_metrics['mse'] += batch_mse * batch_size
            batch_ssim = ssim(out, hr)
            test_metrics['ssims'] += batch_ssim * batch_size
            test_metrics['psnr'] = 10 * log10(1 / (test_metrics['mse'] / test_metrics['nsamples']))
            test_metrics['ssim'] = test_metrics['ssims'] / test_metrics['nsamples']
            tqdm(carn_loader, desc='CARN Predicting: ').set_description(
                desc=f"[Predicting in Test set] PSNR: {test_metrics['psnr']:.4f} dB SSIM: {test_metrics['ssim']:.4f}")

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    carn_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return carn_hics


def save_data(carn, compact, size, file):
    carn = spreadM(carn, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, deephic=carn, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    res_num = args.resblock
    cuda = args.cuda
    print('WARNING: Predict process requires large memory, thus ensure that your machine has ~150G memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    in_dir = os.path.join(root_dir, '../data')
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]
    # deephic_file = [f for f in files if f.find(cell_line.lower() + '.npz') >= 0][0]
    HiCARN_file = 'deephic_10kb40kb_c40_s40_b201_nonpool_human_K562.npz'

    chunk, stride, bound, scale = filename_parser(HiCARN_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data[HiCARN]: {HiCARN_file}')
    carn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
    carn_loader = dataloader(carn_data)

    indices, compacts, sizes = data_info(carn_data)
    carn_hics = deephic_predictor(carn_loader, ckpt_file, scale, res_num, device)


    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(carn_hics[key], compacts[key], sizes[key], file)


    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
