import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def heatmap(ax, mat, title=None, x_label=None, y_label=None, show_bar=True, close_ticks=False):
    cmap = "Reds"
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


# CARN = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_1_Predict/GM12878/predict_chr4_40kb.npz')
# CARN = np.array(CARN['deephic'])[4080:4150, 4080:4150]
# deep = np.load('./Datasets_NPZ/DeepHiC_Predict/GM12878/predict_chr4_40kb.npz')
# deep = np.array(deep['deephic'])[4080:4150, 4080:4150]
# hicnn = np.load("/Users/parkerhicks/Desktop/Datasets_NPZ/HiCNN_Predict/GM12878/predict_chr4_40kb.npz")
# hicnn = np.array(hicnn['deephic'])[4080:4150, 4080:4150]
# hicplus = np.load('./Datasets_NPZ/HiCPlus_Predict/GM12878/predict_chr4_40kb.npz')
# hicplus = np.array(hicplus['deephic'])[4080:4150, 4080:4150]
hicsr = np.load('./Datasets_NPZ/HiCSR_Predict/GM12878/predict_chr4_40kb.npz')
hicsr = np.array(hicsr['deephic'])[4080:4150, 4080:4150]
# PCARN = np.load('./Datasets_NPZ/HiCARN_2_Predict/GM12878/predict_chr4_40kb.npz')
# PCARN = np.array(PCARN['deephic'])[4080:4150, 4080:4150]
# real = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr4_10kb.npz')
# real = (np.array(real['hic'])[4080:4150, 4080:4150]) / 255
fake = np.load('./Datasets_NPZ/mat/GM12878/chr4_40kb.npz')
fake = (np.array(fake['hic'])[4080:4150, 4080:4150]) / 255
# data = [hicnn[4][4000:4250, 4000:4250]]


data = [hicsr, fake]
hic_heatmap(data, dediag=0, ncols=2)
plt.show()
