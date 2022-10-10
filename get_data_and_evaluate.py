import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, reorient_btensors, readfile_btens
from Definitions import cov_mat, voigt_notation, dtd_cov_1d_data2fit_v1
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit


# scripts are in a folder -> change to the folder or directory with the data sets
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")


'load data '

# data and btensors are in scanning order
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')
mask, affine = load_data('mask_pad.nii')

data=data_load[:, :, :, :301]

btensors = readfile_btens('btens_oneB0.txt')
btensors = btensors[:301] * 10**(-3)


'DTI cov-Fit'

# we want to do something for each voxel in the mask and save it
# create empty array of the same physical size as data
# and with "K" values in the 4th dimension for our "K" parameters
K = 28 # number of variables that the fitfunktion has as uotput (from linear least squares fit)
results = np.zeros(data.shape[:3] + (K,))
print('Fit results')

for xyz in np.ndindex(mask.shape):  # loop in N-dimension, xyz is a tuple (x,y,z)
    if mask[xyz]:  # if in mask
        #results[xyz] = dtd_cov_1d_data2fit_v1(data[xyz], btensors, cond_limit=1e-10, clip_eps=1e-16) # fit
        results[xyz] = dtd_cov_1d_data2fit(data[xyz], btensors, cond_limit=1e-20, clip_eps=1e-20)

#nib.Nifti1Image(results, affine).to_filename('results.nii')

# now: fit all of the values
V_MD2_fit = np.zeros(data.shape[:3])
V_iso2_fit = np.zeros(data.shape[:3])
V_shear2_fit = np.zeros(data.shape[:3])
MD_fit = np.zeros(data.shape[:3])
FA_fit = np.zeros(data.shape[:3])
V_MD_fit = np.zeros(data.shape[:3])
V_iso_fit = np.zeros(data.shape[:3])
V_MD1_fit = np.zeros(data.shape[:3])
V_iso1_fit = np.zeros(data.shape[:3])
V_shear_fit = np.zeros(data.shape[:3])
V_shear1_fit = np.zeros(data.shape[:3])
C_MD_fit = np.zeros(data.shape[:3])
C_mu_fit = np.zeros(data.shape[:3])
C_M_fit = np.zeros(data.shape[:3])
C_c_fit = np.zeros(data.shape[:3])
MKi_fit = np.zeros(data.shape[:3])
MKa_fit = np.zeros(data.shape[:3])
MKt_fit = np.zeros(data.shape[:3])
MKad_fit = np.zeros(data.shape[:3])
MK_fit = np.zeros(data.shape[:3])
MKd_fit = np.zeros(data.shape[:3])
uFA_fit = np.zeros(data.shape[:3])
S_I_fit = np.zeros(data.shape[:3])
S_A_fit = np.zeros(data.shape[:3])

print('calculate parameters')

for xyz in np.ndindex(results.shape[:3]):
    # get the ordered solution of the fit
    s0_convfit, d2_convfit, c4_convfit = convert_m(results[xyz])
    # get any other parameters from the fit
    V_MD2_fit[xyz], V_iso2_fit[xyz], V_shear2_fit[xyz], MD_fit[xyz], FA_fit[xyz], V_MD_fit[xyz], V_iso_fit[xyz], V_MD1_fit[xyz], V_iso1_fit[xyz], V_shear_fit[xyz], V_shear1_fit[xyz], C_MD_fit[xyz], \
    C_mu_fit[xyz], C_M_fit[xyz], C_c_fit[xyz], MKi_fit[xyz], MKa_fit[xyz], MKt_fit[xyz], MKad_fit[xyz], MK_fit[xyz], MKd_fit[xyz], uFA_fit[xyz], S_I_fit[xyz], S_A_fit[xyz] = decode_m(
        d2_convfit, c4_convfit, reg=1e-20)

print('save results')

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg/results_dotcorr_biexp_nolin5_new")

# save back nifti of result with same affine as "output_filename"
nib.Nifti1Image(V_MD2_fit, affine).to_filename('V_MD2_fit.nii')
nib.Nifti1Image(V_iso2_fit, affine).to_filename('V_iso2_fit.nii')
nib.Nifti1Image(V_shear2_fit, affine).to_filename('V_shear2_fit.nii')
nib.Nifti1Image(MD_fit, affine).to_filename('MD_fit.nii')
nib.Nifti1Image(FA_fit, affine).to_filename('FA_fit.nii')
nib.Nifti1Image(V_MD_fit, affine).to_filename('V_MD_fit.nii')
nib.Nifti1Image(V_iso_fit, affine).to_filename('V_iso_fit.nii')
nib.Nifti1Image(V_MD1_fit, affine).to_filename('V_MD1_fit.nii')
nib.Nifti1Image(V_iso1_fit, affine).to_filename('V_iso1_fit.nii')
nib.Nifti1Image(V_shear_fit, affine).to_filename('V_shear_fit.nii')
nib.Nifti1Image(V_shear1_fit, affine).to_filename('V_shear1_fit.nii')
nib.Nifti1Image(C_MD_fit, affine).to_filename('C_MD_fit.nii')
nib.Nifti1Image(C_mu_fit, affine).to_filename('C_mu_fit.nii')
nib.Nifti1Image(C_M_fit, affine).to_filename('C_M_fit.nii')
nib.Nifti1Image(C_c_fit, affine).to_filename('C_c_fit.nii')
nib.Nifti1Image(MKi_fit, affine).to_filename('MKi_fit.nii')
nib.Nifti1Image(MKa_fit, affine).to_filename('MKa_fit.nii')
nib.Nifti1Image(MKt_fit, affine).to_filename('MKt_fit.nii')
nib.Nifti1Image(MKad_fit, affine).to_filename('MKad_fit.nii')
nib.Nifti1Image(MK_fit, affine).to_filename('MK_fit.nii')
nib.Nifti1Image(MKd_fit, affine).to_filename('MKd_fit.nii')
nib.Nifti1Image(uFA_fit, affine).to_filename('uFA_fit.nii')
nib.Nifti1Image(S_I_fit, affine).to_filename('S_I_fit.nii')
nib.Nifti1Image(S_A_fit, affine).to_filename('S_A_fit.nii')

