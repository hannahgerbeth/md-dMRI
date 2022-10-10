import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, readfile_btens
from Definitions import dtd_cov_1d_data2fit_v1,  fit_signal_ens, voigt_2_tensor
from Tensor_math_MPaquette import tp
from dtd_cov_MPaquette import convert_m


os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
# data shape (90,60,130,331)
# wm voxel [40, 35, 40, :]
# gm voxel [50, 43, 60, :]
# gm mask [40, 41, 57, :]


' Get data '
data,affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii') #shape (90,60,130,331)
plt.imshow(data[:, :, 65, 1])
plt.show()

' Get B-Tensors '
btensors = readfile_btens('btens_oneB0.txt')
btensors = btensors * 10**(-3) # in ms/µm^2

' Get WM and GM masks '
wm_mask, affine = load_data('WM_mask_220422_final.nii')
gm_mask, affine = load_data('GM_mask_220422_final.nii')

data_wm = data * wm_mask[:, :, :, None]
data_gm = data * gm_mask[:, :, :, None]

K = 28 # number of variables that the fitfunktion has as uotput (from linear least squares fit)

' Fit for White Matter '
results_wm = np.zeros(data.shape[:3] + (K,))
s0_wm = np.zeros(data.shape[:3])
d2_wm = np.zeros((data.shape[:3] + (6,)))
c4_wm = np.zeros(data.shape[:3] + (6, 6))
for xyz in np.ndindex(wm_mask.shape):  # loop in N-dimension, xyz is a tuple (x,y,z)
    if wm_mask[xyz]:  # if in mask
        results_wm[xyz] = dtd_cov_1d_data2fit_v1(data_wm[xyz], btensors, cond_limit=1e-10, clip_eps=1e-16) # fit
        s0_wm[xyz], d2_wm[xyz,:], c4_wm[xyz,:,:] = convert_m(results_wm[xyz])


' Fit for Gray Matter '
results_gm = np.zeros(data.shape[:3] + (K,))
s0_gm = np.zeros(data.shape[:3])
d2_gm = np.zeros((data.shape[:3] + (6,)))
c4_gm = np.zeros(data.shape[:3] + (6, 6))
for xyz in np.ndindex(gm_mask.shape):  # loop in N-dimension, xyz is a tuple (x,y,z)
    if gm_mask[xyz]:  # if in mask
        results_gm[xyz] = dtd_cov_1d_data2fit_v1(data_gm[xyz], btensors, cond_limit=1e-10, clip_eps=1e-16)  # fit
        s0_gm[xyz], d2_gm[xyz,:], c4_gm[xyz,:,:] = convert_m(results_gm[xyz])



' Generate data from the fit values: cumulant fit '
data_cumfit_wm = np.zeros(data.shape)
for xyz in np.ndindex(wm_mask.shape):
    if wm_mask[xyz]:
        data_cumfit_wm[xyz] = fit_signal_ens(btensors, d2_wm[xyz], c4_wm[xyz])


data_cumfit_gm = np.zeros(data.shape)
for xyz in np.ndindex(gm_mask.shape[:3]):
    if gm_mask[xyz]:
        data_cumfit_gm[xyz] = fit_signal_ens(btensors, d2_gm[xyz], c4_gm[xyz])



' Generate data from fit values: simple signal fit '


def exp_signal(btens, d2):
    temp = np.zeros(btens.shape[0] )
    for i in range(btens.shape[0]):
        temp[i] = np.exp(- tp(btens[i], voigt_2_tensor(d2)))
    return temp


data_fit_wm = np.zeros(data.shape)
for xyz in np.ndindex(wm_mask.shape):
    if wm_mask[xyz]:
        data_fit_wm[xyz] = exp_signal(btensors, d2_wm[xyz])


data_fit_gm = np.zeros(data.shape)
for xyz in np.ndindex(gm_mask.shape):
    if gm_mask[xyz]:
        data_fit_gm[xyz] = exp_signal(btensors, d2_gm[xyz])



' Plots '

plt.plot(np.mean(data_wm[np.where(wm_mask == 1)], axis=(0)), label='Signal averaged over WM ROI')
plt.plot(np.mean(data_cumfit_wm[np.where(wm_mask == 1)], axis=(0)), label='Signal fitted: cumulant expansion ')
plt.plot(np.mean(data_fit_wm[np.where(wm_mask == 1)], axis=(0)), label='Signal fitted: simple exponential')
plt.title('White matter')
plt.legend()
plt.show()

plt.plot(np.mean(data_gm[np.where(gm_mask == 1)], axis=(0)), label='Signal averaged over GM ROI')
#plt.plot(np.mean(data_cumfit_gm[np.where(gm_mask == 1)], axis=(0)), label='Signal fitted: cumulant expansion')
plt.plot(np.clip(np.mean(data_cumfit_gm[np.where(gm_mask == 1)], axis=(0)), 0, 1), label='Signal fitted: cumulant expansion')
plt.plot(np.mean(data_fit_gm[np.where(gm_mask == 1)], axis=(0)), label='Signal fitted: simple exponential')
plt.title('Gray matter')
plt.legend()
plt.show()

