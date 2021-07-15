import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

def heatmap(ax, mat, title=None, x_label=None, y_label=None, show_bar=True, close_ticks=False):
    cmap = "Reds"
    # vmin, vmax = np.nanmin(mat), np.nanmax(mat) # get the max/min value and ignore nan
    vmin, vmax = 0, 0.1
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
        nrows = int(len(data)//ncols+1)
        figure = plt.figure(facecolor='w', figsize=(4.9*ncols, 4*nrows))
        gs = gridspec.GridSpec(nrows, ncols)
        for i, mat in enumerate(data):
            row, col = i // ncols, i % ncols
            axs.append(figure.add_subplot(gs[row, col]))
            if dediag > 0 and mat.ndim==2:
                mat = np.triu(mat, dediag) + np.triu(mat.T, dediag).T
            # only show details on the first row and the first column
            title = titles[col] if titles is not None else None
            y_label = y_labels[row] if col==0 and y_labels is not None else None
            x_label = x_labels[col] if row==0 and x_labels is not None else None
            heatmap(axs[-1], mat, title, x_label, y_label)
    else:
        figure = plt.figure(facecolor='w')
        ax = figure.add_subplot(1,1,1)
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
    idx_x = np.where((x>x.min()) & (x<x.max()))[0]
    idx_y = np.where((y>y.min()) & (y<y.max()))[0]
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
    # g = g.annotate(stats.pearsonr)
    return data

import torch
from torch.utils.data import TensorDataset, DataLoader
from skimage.util import img_as_float

# valid_file = '/Users/parkerhicks/Desktop/Datasets_NPZ/data/deephic_10kb40kb_c40_s40_b201_nonpool_valid.npz'
# valid = np.load(valid_file)
#
# valid_data = torch.tensor(valid['data'], dtype=torch.float)
# valid_target = torch.tensor(valid['target'], dtype=torch.float)
# valid_inds = torch.tensor(valid['inds'], dtype=torch.long)
#
# valid_set = TensorDataset(valid_data, valid_target, valid_inds)
#
# valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, drop_last=True)

# HR = np.load('./Datasets_NPZ/mat/K562/chr19_10kb.npz')
# HR = np.array(HR['hic'])
# LR = np.load('./Datasets_NPZ/mat/K562/chr19_40kb.npz')
# LR = np.array(LR['hic'])

CARN = np.load('./Datasets_NPZ/CARN_predict/Recent/K562/predict_chr11_40kb_40.npz')
CARN = np.array(CARN['deephic'])
deep = np.load('./Datasets_NPZ/DeepHiC_predict/server_predict_K562_chr11_40kb_40.npz')
deep = np.array(deep['deephic'])
# hicsr = np.load('./Datasets_NPZ/HiCSR_Predict/predict_chr4_40kb.npz')
# hicsr = np.array(hicsr['deephic'])
# CARN_deep = np.load('./Datasets_NPZ/CARN_Deep_Predict/Recent/Gm12878/predict_chr4_40kb_40.npz')
# CARN_deep = np.array(CARN_deep['deephic'])
PCARN = np.load('./Datasets_NPZ/PCARN_Predict/Recent/K562/predict_chr11_40kb_40.npz')
PCARN = np.array(PCARN['deephic'])
# real = np.load('./Datasets_NPZ/mat/GM12878/chr4_10kb.npz')
# real = np.array(real['hic'])
# real_max = np.max(real)
# fake = real / 16

data = [CARN[5500:6000, 5500:6000], deep[5500:6000, 5500:6000], PCARN[5500:6000, 5500:6000]]
hic_heatmap(data, dediag=0, ncols=3)
plt.show()

