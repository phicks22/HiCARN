import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from pathlib import Path


# def _toimg(mat):
#     m = torch.tensor(mat)
#     # convert to float and add channel dimension
#     return m.float()
#
# def _tohic(mat):
#     mat.squeeze_()
#     return mat.numpy()#.astype(int)
#
# def gaussian(width, sigma):
#     gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
#     return gauss / gauss.sum()
#
# def create_window(window_size, channel, sigma=3):
#     _1D_window = gaussian(window_size, sigma).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window
#
# def gaussian_filter(img, width, sigma=3):
#     img = _toimg(img).unsqueeze(0)
#     _, channel, _, _ = img.size()
#     window = create_window(width, channel, sigma)
#     mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
#     return _tohic(mu1)
#
# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# def ssim(img1, img2, window_size=11, size_average=True):
#     img1 = _toimg(img1).unsqueeze(0)
#     img2 = _toimg(img2).unsqueeze(0)
#     channel = img1.size()[1]
#     window = create_window(window_size, channel)
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)
#
# class SSIM(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     def forward(self, img1, img2):
#         channel = img1.size()[1]
#
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
#
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#
#             self.window = window
#             self.channel = channel
#
#         return print(_ssim(img1, img2, window, self.window_size, channel, self.size_average))
#
# def noise_estimator(mat):
#     # https://www.cnblogs.com/algorithm-cpp/p/4105943.html
#     kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
#     img = _toimg(mat).unsqueeze(0)
#     ker = _toimg(kernel).unsqueeze(0)
#     out = F.conv2d(img, ker)
#     out = _tohic(out)
#     noise = np.sum(out)/(out.shape[0]*out.shape[1])
#     return noise
    
#
# HR = np.load('./Datasets_NPZ/mat/GM12878/chr16_10kb.npz')
# CARN = np.load('./Datasets_NPZ/CARN_predict/GM12878/predict_chr16_40kb.npz')
# OG = np.load('./Datasets_NPZ/mat/GM12878/chr16_40kb.npz')
# deep = np.load('./Datasets_NPZ/DeepHiC_Predict/GM12878/predict_chr16_40kb.npz')
# PCARN = np.load('./Datasets_NPZ/PCARN_Predict/GM12878/predict_chr16_40kb.npz')
# CARN_deep = np.load('./Datasets_NPZ/CARN_Deep_Predict/GM12878/chr15_22/predict_chr16_40kb.npz')
#
#
# HR = np.array(HR['hic'])
# CARN = np.array(CARN['deephic'])
# OG = np.array(OG['hic'])
# deep = np.array(deep['deephic'])
# PCARN = np.array(PCARN['deephic'])
# CARN_deep = np.array(CARN_deep['deephic'])
#
#
# HR = torch.tensor(HR, dtype=torch.float)
# CARN = torch.tensor(CARN, dtype=torch.float)
# OG = torch.tensor(OG, dtype=torch.float)
# deep = torch.tensor(deep, dtype=torch.float)
# PCARN = torch.tensor(PCARN, dtype=torch.float)
# CARN_deep = torch.tensor(CARN_deep, dtype=torch.float)

# HR = HR.unsqueeze(0)
# CARN = CARN.unsqueeze(0)
# OG = OG.unsqueeze(0)
# deep = deep.unsqueeze(0)
# PCARN = PCARN.unsqueeze(0)
# CARN_deep = CARN_deep.unsqueeze(0)
#
# # HR = _toimg(HR)
# # SR = _toimg(SR)
#
# # print("SSIM: ", ssim(HR, SR))
# # print("MSE: ", ((SR - HR)**2).mean())
#
#
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from math import log10
#
# HR = img_as_float(HR)
# CARN = img_as_float(CARN)
# OG = img_as_float(OG)
# deep = img_as_float(deep)
# PCARN = img_as_float(PCARN)
# CARN_deep = img_as_float(CARN_deep)
# #
# rows, cols = CARN.shape

# noise = np.ones_like(SR) * 0.2 * (SR.max() - SR.min())
# rng = np.random.default_rng()
# noise[rng.random(size=noise.shape) > 0.5] *= -1
#
# img_noise = SR + noise
# img_const = SR + abs(noise)

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 4),
#                          sharex="none", sharey="none")
# ax = axes.ravel()
#
# mse_none = mean_squared_error(HR, HR)
# ssim_none = ssim(HR, HR, data_range=HR.max() - HR.min())
#
# mse_CARN = mean_squared_error(HR, CARN)
# ssim_CARN = ssim(HR, CARN, data_range=CARN.max() - CARN.min())
# psnr_CARN = 10 * log10(1 / mse_CARN)
#
# mse_OG = mean_squared_error(HR, OG)
# ssim_OG = ssim(HR, OG, data_range=OG.max() - OG.min())
# psnr_OG = 10 * log10(1 / mse_OG)
#
# mse_deep = mean_squared_error(HR, deep)
# ssim_deep = ssim(HR, deep, data_range=deep.max() - deep.min())
# psnr_deep = 10 * log10(1 / mse_deep)

# mse_PCARN = mean_squared_error(HR, PCARN)
# ssim_PCARN = ssim(PCARN, HR, data_range=PCARN.max() - PCARN.min())
# psnr_PCARN = 10 * log10(1 / mse_PCARN)
#
# mse_CARN_deep = mean_squared_error(HR, CARN_deep)
# ssim_CARN_deep = ssim(CARN_deep, HR, data_range=CARN_deep.max() - CARN_deep.min())
# psnr_CARN_deep = 10 * log10(1 / mse_CARN_deep)

# print("SSIM HR 10kb: ", ssim_none)
# print("MSE HR 10kb: ", mse_none)
# print("SSIM LR 40kb: ", ssim_OG)
# print("MSE LR 40kb: ", mse_OG)
# print("PSNR LR 40kb: ", psnr_OG)
# print("SSIM CARN-only: ", ssim_CARN)
# print("MSE CARN-only: ", mse_CARN)
# print("PSNR CARN-only: ", psnr_CARN)
# print("SSIM DeepHiC: ", ssim_deep)
# print("MSE DeepHiC: ", mse_deep)
# print("PSNR DeepHiC: ", psnr_deep)
# print("SSIM PCARN: ", ssim_PCARN)
# print("MSE PCARN: ", mse_PCARN)
# print("PSNR PCARN: ", psnr_PCARN)
# print("SSIM CARN-Deep: ", ssim_CARN_deep)
# print("MSE CARN-Deep: ", mse_CARN_deep)
# print("PSNR CARN-Deep: ", psnr_CARN_deep)

# print("SSIM HR 10kb: ", ssim(HR, HR))
# # print("MSE HR 10kb: ", mse_none)
# print("SSIM LR 40kb: ", ssim(HR, (OG * 1/16)))
# # print("MSE LR 40kb: ", mse_OG)
# # print("PSNR LR 40kb: ", psnr_OG)
# print("SSIM CARN-only: ", ssim(HR, CARN))
# # print("MSE CARN-only: ", mse_CARN)
# # print("PSNR CARN-only: ", psnr_CARN)
# print("SSIM DeepHiC: ", ssim(HR, deep))
# # print("MSE DeepHiC: ", mse_deep)
# # print("PSNR DeepHiC: ", psnr_deep)
# print("SSIM PCARN: ", ssim(HR, PCARN))
# # print("MSE PCARN: ", mse_PCARN)
# # print("PSNR PCARN: ", psnr_PCARN)
# print("SSIM CARN-Deep: ", ssim(HR, CARN_deep))
# # print("MSE CARN-Deep: ", mse_CARN_deep)
# # print("PSNR CARN-Deep: ", psnr_CARN_deep)


# vmin_HR = np.nanmin(HR)
# vmax_HR = np.nanmax(HR)
#
# vmin_OG = np.nanmin(OG)
# vmax_OG = np.nanmax(OG)
#
# vmin_CARN = np.nanmin(CARN)
# vmax_CARN = np.nanmax(CARN)
#
# vmin_deep = np.nanmin(deep)
# vmax_deep = np.nanmax(deep)
#
# vmin_PCARN = np.nanmin(PCARN)
# vmax_PCARN = np.nanmax(PCARN)
#
# vmin_CARN_deep = np.nanmin(CARN_deep)
# vmax_CARN_deep = np.nanmax(CARN_deep)

# mse_noise = mean_squared_error(SR, img_noise)
# ssim_noise = ssim(SR, img_noise,
#                   data_range=img_noise.max() - img_noise.min())
#
# mse_const = mean_squared_error(SR, img_const)
# ssim_const = ssim(SR, img_const,
#                   data_range=img_const.max() - img_const.min())

# label = 'MSE: {:.2f}, SSIM: {:.2f}'

# ax[0].imshow(HR, cmap='Reds', vmin=0, vmax=1)
# ax[0].set_xlabel(label.format(mse_none, ssim_none))
# ax[0].set_title('10kb Resolution')
#
# ax[1].imshow(OG, cmap='Reds', vmin=0, vmax=1)
# ax[1].set_xlabel(label.format(mse_OG, ssim_OG))
# ax[1].set_title('40kb Resolution')
#
# ax[2].imshow(CARN[40:140, 40:140], cmap='Reds', vmin=0, vmax=1)
# ax[2].set_xlabel(label.format(mse_CARN, ssim_CARN))
# ax[2].set_title('CARN-only (VEHiCLE dataset, 100 epochs)')
#
# ax[3].imshow(deep[40:140, 40:140], cmap='Reds', vmin=0, vmax=1)
# ax[3].set_xlabel(label.format(mse_deep, ssim_deep))
# ax[3].set_title('DeepHiC')
# #
# ax[2].imshow(PCARN[40:140, 40:140], cmap='Reds', vmin=vmin_PCARN, vmax=vmax_PCARN)
# ax[2].set_xlabel(label.format(mse_PCARN, ssim_PCARN))
# ax[2].set_title('PCARN (Ours)')
# #
# ax[3].imshow(CARN_deep[40:140, 40:140], cmap='Reds', vmin=vmin_CARN_deep, vmax=vmax_CARN_deep)
# ax[3].set_xlabel(label.format(mse_CARN_deep, ssim_CARN_deep))
# ax[3].set_title('CARN-Deep (Ours)')
#
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def heatmap(ax, mat, title=None, x_label=None, y_label=None, show_bar=True, close_ticks=False):
    cmap = "Reds"
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)  # get the max/min value and ignore nan
    im = ax.matshow(mat, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    if show_bar:
        cbar = plt.colorbar(im, ax=ax, aspect=9, shrink=0.3, ticks=[vmin, vmax])
        cbar.ax.yaxis.set_ticks_position('none')
        cbar.ax.set_yticklabels([])
        cbar.ax.set_xlabel('Low', fontsize='large')
        cbar.ax.set_title('High', fontsize='large')
        cbar.ax.set_ylabel('reads', rotation=-90, va='bottom', fontsize='x-large')
    if close_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)


def hic_heatmap(data, dediag=0, ncols=4, titles=None, x_labels=None, y_labels=None, file=None):
    if isinstance(data, list):
        axs = []
        nrows = int(len(data) // ncols + 1)
        figure = plt.figure(facecolor='w', figsize=(4.9 * ncols, 4 * nrows))
        gs = gridspec.GridSpec(nrows, ncols)
        for i, mat in enumerate(data):
            row, col = i // ncols, i % ncols
            axs.append(figure.add_subplot(gs[row, col]))
            if dediag > 0 and mat.ndim == 2:
                mat = np.triu(mat, dediag) + np.triu(mat.T, dediag).T
            # only show details on the first row and the first column
            title = titles[col] if titles is not None else None
            y_label = y_labels[row] if col == 0 and y_labels is not None else None
            x_label = x_labels[col] if row == 0 and x_labels is not None else None
            heatmap(axs[-1], mat, title, x_label, y_label)
    else:
        figure = plt.figure(facecolor='w')
        ax = figure.add_subplot(1, 1, 1)
        if dediag > 0:
            data = np.triu(data, dediag) + np.triu(data.T, dediag).T
        heatmap(ax, data, title=titles, x_label=x_labels, y_label=y_labels)
    figure.tight_layout()
    if file is not None:
        figure.savefig(file, format='svg')


def surf(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    m, n = data.shape
    x, y = np.meshgrid(range(m), range(n))
    ax.plot_surface(x, y, data)


import pandas as pd
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def _clear_max_min(x, y):
    idx_x = np.where((x > x.min()) & (x < x.max()))[0]
    idx_y = np.where((y > y.min()) & (y < y.max()))[0]
    idx_setx, idx_sety = set(idx_x), set(idx_y)
    inter_idx = np.array(list(idx_setx.intersection(idx_sety)))
    return x[inter_idx], y[inter_idx]


def hic_joint(mat, rep, distance=(2, 101), clear_max_min=False):
    x, y = [], []
    for i in range(*distance):
        diagx, diagy = np.diag(mat, k=i), np.diag(rep, k=i)
        x.append(diagx)
        y.append(diagy)
    x = np.concatenate(x)
    y = np.concatenate(y)
    if (clear_max_min):
        x, y = _clear_max_min(x, y)
    data = pd.DataFrame({'origin': x, 'replication': y})
    g = sns.JointGrid(x='origin', y='replication', data=data)
    g = g.plot_joint(plt.scatter, s=40, edgecolor='white')
    g = g.plot_marginals(sns.distplot, kde=True)
    g = g.annotate(stats.pearsonr)
    return data

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from CARN import Generator
from models.loss import GeneratorLoss

batch_size = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netG = Generator(num_channels=64, scale_factor=1)
criterionG = GeneratorLoss().to(device)


valid_file = '/Users/parkerhicks/Desktop/Datasets_NPZ/data/deephic_10kb40kb_c40_s40_b201_nonpool_valid.npz'
valid = np.load(valid_file)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)


valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}

best_ssim = 0


valid_bar = tqdm(valid_loader)
with torch.no_grad():
    for val_lr, val_hr, inds in valid_bar:
        batch_size = val_lr.size(0)
        valid_result['nsamples'] += batch_size
        lr = val_lr.to(device)
        hr = val_hr.to(device)
        sr = netG(lr)

        sr_out = sr
        hr_out = hr
        g_loss = criterionG(sr_out.mean(), sr, hr)

        valid_result['g_loss'] += g_loss.item() * batch_size

        batch_mse = ((sr - hr) ** 2).mean()
        valid_result['mse'] += batch_mse * batch_size
        batch_ssim = ssim(sr, hr)
        valid_result['ssims'] += batch_ssim * batch_size
        valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
        valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
        valid_bar.set_description(
            desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
valid_gscore = valid_result['g_score'] / valid_result['nsamples']
now_ssim = valid_result['ssim'].item()


if now_ssim > best_ssim:
    best_ssim = now_ssim
    print(f'Now, Best ssim is {best_ssim:.6f}')

