import numpy as np
import os

npy_file_fake = (np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/HiCARN_1_Predict/MAE_Loss/GM12878/predict_chr4_40kb.npz'))
# npy_file_real = (np.load('/Users/parkerhicks/Desktop/Datasets_NPZ/mat/GM12878/chr4_10kb.npz'))

npy_array_fake = np.array(npy_file_fake['deephic'])[4000:4250, 4200:4250]
# npy_array_real = np.array(npy_file_real['hic'])

npy_txt_fake = np.savetxt(fname='GM12878_Chr_4_HiCARN_1_MAE_Loss_40Mb_42_5Mb', X=npy_array_fake)
# npy_txt_real = np.savetxt(fname='GM12878_Chr_4_Real', X=npy_array_real)

# os.path.join(os.path.expanduser('~'), 'Desktop', npy_txt_fake)
# os.path.join(os.path.expanduser('~'), 'Desktop', npy_txt_real)
