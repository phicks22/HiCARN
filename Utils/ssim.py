import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def _toimg(mat):
    m = torch.tensor(mat)
    # convert to float and add channel dimension
    return m.float()

def _tohic(mat):
    mat.squeeze_()
    return mat.numpy()#.astype(int)

def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
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
    img1 = _toimg(img1).unsqueeze(0)
    img2 = _toimg(img2).unsqueeze(0)
    channel = img1.size()[1]
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        img1 = _toimg(img1).unsqueeze(0)
        img2 = _toimg(img2).unsqueeze(0)
        channel = img1.size()[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return print(_ssim(img1, img2, window, self.window_size, channel, self.size_average))
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
HR = np.load('./Datasets_NPZ/mat/GM12878/chr16_10kb.npz')
CARN = np.load('./Datasets_NPZ/CARN_predict/GM12878/predict_chr16_40kb.npz')
OG = np.load('./Datasets_NPZ/mat/GM12878/chr16_40kb.npz')
deep = np.load('./Datasets_NPZ/DeepHiC_Predict/GM12878/predict_chr16_40kb.npz')
# PCARN = np.load('./Datasets_NPZ/PCARN_Predict/GM12878/predict_chr16_40kb.npz')
# CARN_deep = np.load('./Datasets_NPZ/CARN_Deep_Predict/GM12878/chr15_22/predict_chr16_40kb.npz')


HR = np.array(HR['hic'])
CARN = np.array(CARN['deephic'])
OG = np.array(OG['hic'])
deep = np.array(deep['deephic'])
# PCARN = np.array(PCARN['deephic'])
# CARN_deep = np.array(CARN_deep['deephic'])
#
#
HR = torch.tensor(HR, dtype=torch.float)
CARN = torch.tensor(CARN, dtype=torch.float)
OG = torch.tensor(OG, dtype=torch.float)
deep = torch.tensor(deep, dtype=torch.float)
# PCARN = torch.tensor(PCARN, dtype=torch.float)
# CARN_deep = torch.tensor(CARN_deep, dtype=torch.float)

HR = HR.unsqueeze(0)
CARN = CARN.unsqueeze(0)
OG = OG.unsqueeze(0)
deep = deep.unsqueeze(0)
# PCARN = PCARN.unsqueeze(0)
# CARN_deep = CARN_deep.unsqueeze(0)
#
# # HR = _toimg(HR)
# # SR = _toimg(SR)
# output   = torch.zeros((1,1,269,269))
# for i in range(0, 269-40, 28):
#     for j in range(0,269-40,28):
#         temp = HR[:,:,i:i+40, j:j+40]
#         output[:,:,i+6:i+34, j+6:j+34] = model(temp)
# output = output[:,:,6:-6,6:-6].detach()



example1 = SSIM()
print(example1.forward(HR, CARN))

example2 = SSIM()
print(example2.forward(HR, deep))
#
#
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
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
