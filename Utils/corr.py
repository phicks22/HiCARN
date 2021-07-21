import numpy as np
from scipy.stats import pearsonr, spearmanr

def _clear_max_min(x, y):
    idx_x = set(np.where((x>x.min()) & (x<x.max()))[0])
    idx_y = set(np.where((y>y.min()) & (y<y.max()))[0])
    inter_idx = sorted(list(idx_x.intersection(idx_y)))
    return x[inter_idx], y[inter_idx]

def diagcorr(mat1, mat2, rtype='pearson', max_shift=100, percentile=100, clearmaxmin=False, symmetric=False):
    """ function for calculating pearson correlation along with distance genome. """
    l1, l2 = len(mat1), len(mat2)
    # adjust to same size
    padding = (l1 - l2) // 2
    assert padding >= 0 and (l1-l2)%2 == 0, \
        "The first matrix must be larger than the second one, and padding must be symmetric!"
    if padding > 0:
        mat1 = mat1[padding:l1-padding, padding:l1-padding]

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

    return r, p

mat1 = '/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_1_Predict/MAE_Loss/GM12878/predict_chr14_40kb.npz'
mat = '/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr14_10kb.npz'
mat1 = (np.load(mat1)['deephic'])[2250:2500, 2250:2500]
mat2 = (np.load(mat)['hic'])[2250:2500, 2250:2500]



r_list = []
for i in diagcorr(mat1, mat2):
    r_list += i

pcc = sum(r_list) / len(r_list)

print(pcc)
