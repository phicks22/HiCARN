import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def heatmap(ax, mat, title=None, x_label=None, y_label=None, show_bar=True, close_ticks=False):
    cmap = "Reds"
    # vmin, vmax = np.nanmin(mat), np.nanmax(mat) # get the max/min value and ignore nan
    vmin, vmax = 0, 0.5
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

    return im


def hic_heatmap(data, dediag=0, ncols=1, titles=None, x_labels=None, y_labels=None, file=None):
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


def _clear_max_min(x, y):
    idx_x = np.where((x > x.min()) & (x < x.max()))[0]
    idx_y = np.where((y > y.min()) & (y < y.max()))[0]
    idx_setx, idx_sety = set(idx_x), set(idx_y)
    inter_idx = np.array(list(idx_setx.intersection(idx_sety)))
    return x[inter_idx], y[inter_idx]


CARN = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_1_Predict/10CB_MSE/GM12878/predict_chr14_40kb.npz')
CARN = np.array(CARN['deephic'])[2250:2500, 2250:2500]
deep = np.load('./Datasets_NPZ/DeepHiC_Predict/GM12878/predict_chr14_40kb.npz')
deep = np.array(deep['deephic'])[2250:2500, 2250:2500]
hicnn = np.load("/Users/parkerhicks/Desktop/Datasets_NPZ/HiCNN_Predict/GM12878/predict_chr14_40kb.npz")
hicnn = np.array(hicnn['deephic'])[2250:2500, 2250:2500]
# hicnn = np.load("/Users/parkerhicks/Desktop/Datasets_NPZ/HiCNN2_Predict/GM12878/HiCNN_Predict_chr4_40kb.npy", allow_pickle=True).item()
hicplus = np.load('./Datasets_NPZ/HiCPlus_Predict/GM12878/predict_chr14_40kb.npz')
hicplus = np.array(hicplus['deephic'])[2250:2500, 2250:2500]
# hicsr = np.load('./Datasets_NPZ/HiCSR_Predict/predict_chr4_40kb.npz')
# hicsr = np.array(hicsr['deephic'])[4000:4500, 4000:4500]
# CARN_deep = np.load('./Datasets_NPZ/CARN_Deep_Predict/Recent/Gm12878/predict_chr4_40kb_40.npz')
# CARN_deep = np.array(CARN_deep['deephic'])
# PCARN = np.load('./Datasets_NPZ/PCARN_Predict/Recent/GM12878/predict_chr4_40kb_40.npz')
# PCARN = np.array(PCARN['deephic'])[4000:4500, 4000:4500]
real = np.load('./Datasets_NPZ/mat/GM12878/chr14_10kb.npz')
real = (np.array(real['hic'])[2250:2500, 2250:2500]) / 255
fake = real / 16
# data = [hicnn[4][4000:4250, 4000:4250]]
# hic_heatmap(data, dediag=0, ncols=1)
# plt.show()
# hicnn = hicnn

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}


def divide(mat, chr_num, chunk_size=40, stride=28, bound=201, padding=True, species='hsa', verbose=False):
    chr_str = str(chr_num)
    if isinstance(chr_num, str): chr_num = except_chr[species][chr_num]
    result = []
    index = []
    size = mat.shape[0] // 2
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height // 2, stride):
        for j in range(0, width // 2, stride):
            if abs(i - j) <= bound and i + chunk_size < height and j + chunk_size < width:
                subImage = mat[i:i + chunk_size, j:j + chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(
        f'[Chr{chr_str}] Deviding HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, '
        f'stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    chr_nums = sorted(list(np.unique(indices[:, 0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:, 0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0, 1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[i:i + h, j:j + w] = sub
        results[n] = full_mat
    return results


# divided = divide(hicnn, chr_num=4, chunk_size=28, stride=34, bound=201, padding=True)
# mat = together(divided[0], divided[1])

# num_bins = np.ceil(191154276 / 40000).astype('int')
# mat = np.zeros((num_bins, num_bins))
# for i in range(divided.shape[0]):
#     # r1 = divided[i, 0]
#     # c1 = divided[i, 1]
#     r1 = 28
#     c1 = 28
#     r2 = r1 + 27 + 1
#     c2 = c1 + 27 + 1
#     mat[r1:r2, c1:c2] = divided[i, :, :]
#
# # copy upper triangle to lower triangle
# lower_index = np.tril_indices(num_bins, -1)
# mat[lower_index] = mat.T[lower_index]
data = [CARN, real, hicnn, deep]
hic_heatmap(data, dediag=0, ncols=2)
plt.show()
