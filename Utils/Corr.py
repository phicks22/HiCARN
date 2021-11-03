import numpy as np
import os
from scipy.stats import pearsonr, spearmanr


def _clear_max_min(x, y):
    idx_x = set(np.where((x > x.min()) & (x < x.max()))[0])
    idx_y = set(np.where((y > y.min()) & (y < y.max()))[0])
    inter_idx = sorted(list(idx_x.intersection(idx_y)))
    return x[inter_idx], y[inter_idx]


def diagcorr(mat1, mat2, rtype='pearson', max_shift=100, percentile=100, clearmaxmin=False, symmetric=False):
    """
    Function for calculating pearson correlation along with distance genome.
    """
    l1, l2 = len(mat1), len(mat2)
    # adjust to same size
    padding = (l1 - l2) // 2
    assert padding >= 0 and (l1 - l2) % 2 == 0, \
        "The first matrix must be larger than the second one, and padding must be symmetric!"
    if padding > 0:
        mat1 = mat1[padding:l1 - padding, padding:l1 - padding]

    assert l2 > max_shift, "Shifting distance is too large for input matrices!"

    r = np.zeros(max_shift)
    p = np.zeros(max_shift)
    for s in range(max_shift):
        diag1 = np.diag(mat1, k=s)
        diag2 = np.diag(mat2, k=s)
        if symmetric:
            diag1 = (diag1 + np.diag(mat1, k=-s)) / 2
            diag2 = (diag2 + np.diag(mat2, k=-s)) / 2

        if percentile < 100:
            diag1 = np.minimum(np.percentile(diag1, percentile), diag1)
            diag2 = np.minimum(np.percentile(diag2, percentile), diag2)

        if clearmaxmin:
            diag1, diag2 = _clear_max_min(diag1, diag2)

        if rtype == 'pearson':
            r[s], p[s] = pearsonr(diag1, diag2)
        elif rtype == 'spearman':
            r[s], p[s] = spearmanr(diag1, diag2)

    return r


# root_pred = '/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_2_Predict/GM12878/5CB/'
# root_real = '/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/'

# for file in os.listdir(root):
#     mat1 = np.load(file)
#     mat1 = mat1['deephic']
#
#     rplist = []
#     for i in diagcorr(mat1, mat2, rtype='pearson'):
#         rplist.append(i)
#
#     rp = sum(rplist) / len(rplist)
#     cor_dict[f'pearson_{file}'] = rp
#
#     rslist = []
#     for i in diagcorr(mat1, mat2, rtype='spearman'):
#         rslist.append(i)
#
#     rs = sum(rslist) / len(rslist)
#     cor_dict[f'spearman_{file}'] = rs

mat1 = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_1_Predict/10CB_MSE/GM12878/predict_chr20_40kb.npz')
mat1 = ((mat1['deephic']))

mat2 = np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr20_10kb.npz')
mat2 = ((mat2['hic']))

rplist = []
for i in diagcorr(mat1, mat2, rtype='pearson'):
    rplist.append(i)

rp = sum(rplist) / len(rplist)

rslist = []
for i in diagcorr(mat1, mat2, rtype='spearman'):
    rslist.append(i)

rs = sum(rslist) / len(rslist)

print("Pearson: ", rp, '\n', "Spearman: ", rs)
