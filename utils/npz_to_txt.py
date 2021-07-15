import numpy as np
import os

npy_file_fake = (np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/CARN_Predict/Recent/GM12878/predict_chr4_40kb_40_usethis.npz'))
npy_file_real = (np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr4_10kb.npz'))

npy_array_fake = np.array(npy_file_fake['deephic'])
npy_array_real = np.array(npy_file_real['hic'])

npy_txt_fake = np.savetxt(fname='GM12878_Chr_4_CARN', X=npy_array_fake)
npy_txt_real = np.savetxt(fname='GM12878_Chr_4_Real', X=npy_array_real)

os.path.join(os.path.expanduser('~'), 'Desktop', npy_txt_fake)
os.path.join(os.path.expanduser('~'), 'Desktop', npy_txt_real)
