import numpy as np
import os
import nibabel as nib
from Definitions_smallscripts import load_data, readfile_btens, mean_Sb
from Definitions_smallscripts import monoexp_fit, biexp_fit
import time


' Load data, btensors, bvalues and mask '
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')  # shape (90,60,130,331)

btensors = readfile_btens('btens_oneB0.txt')
bvals = np.loadtxt('bvals_oneB0.txt')  # bvals in s/mm2

btensors = btensors * 10 ** (-3)  # ms/ym^2
bvals = bvals * 10 ** (-3)

mask, affine = load_data('mask_pad.nii')


# index of the first linear shell
# dot-compartment correction is performed on the linear acquisition with the highest diffusion weighting
lin_idx = 6

b_mean_tmp, S_mean_tmp = mean_Sb(data_load[0,0,0], bvals)
b_mean_lin_data = np.hstack([0, b_mean_tmp[lin_idx:]])

S_mean_lin_data = np.zeros((data_load.shape[0], data_load.shape[1], data_load.shape[2], b_mean_lin_data.shape[0]))
for xyz in np.ndindex(mask.shape):
    if mask[xyz]:
        S_mean_data = mean_Sb(data_load[xyz], bvals)[1]
        S_mean_lin_data[xyz] = np.hstack([S_mean_data[0], S_mean_data[lin_idx:]])


'biexponential fit '
start = time.time()
params_biexp = np.zeros((data_load.shape[0], data_load.shape[1], data_load.shape[2], 4))
for xyz in np.ndindex(mask.shape):
    if mask[xyz]:
        print(xyz)

        # fit monoexp to initialize the cumulant
        params_mono, pcov, init_mono = monoexp_fit(S_mean_lin_data[xyz], b_mean_lin_data)

        # fit cumulant
        params_biexp[xyz], pcov, init_biexp = biexp_fit(S_mean_lin_data[xyz], b_mean_lin_data, params_mono)
print((time.time() - start)/60, 'minutes')

# save the biexponential parameters
#nib.Nifti1Image(params_biexp, affine).to_filename('dot_correction_biexp_params_10000its_fulldata.nii')


epsilon = 1e-6
data_biexp_corrected_clipped = np.zeros((data_load.shape))
for xyz in np.ndindex(mask.shape):
    if mask[xyz]:
        data_biexp_corrected_clipped[xyz] = np.clip((data_load[xyz] - params_biexp[xyz][3]) / (1 - params_biexp[xyz][3]), 0, 1) + epsilon

#nib.Nifti1Image(data_biexp_corrected_clipped, affine).to_filename('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')
