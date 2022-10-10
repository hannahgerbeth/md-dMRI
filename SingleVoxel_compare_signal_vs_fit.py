import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, reorient_btensors, readfile_btens
from Definitions import cov_mat, voigt_notation, dtd_cov_1d_data2fit_v1
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit
from Tensor_math_MPaquette import _S_ens


# scripts are in a folder -> change to the folder or directory with the data sets
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')
data = data_load[40, 35, 40,:301] # wm voxel
#data = data_load[50, 43, 60,:241] # gm voxel

data_load_dc, affine = load_data('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')
data_dc = data_load_dc[40, 35, 40,:301] # wm voxel
#data_dc = data_load_dc[50, 43, 60,:241] # gm voxel

print('load data')
btensors_load = readfile_btens('btens_oneB0.txt')
bvals_load = np.loadtxt('bvals_oneB0.txt')

btensors = btensors_load[:301] * 10**(-3)
bvals = bvals_load[:301]* 10**(-3)

mask, affine = load_data('mask_pad.nii')

' plot pure data '
plt.plot(data, label='data')
plt.plot(data_dc, label='data_dc')
plt.legend()
plt.show()



# fit the normal data
results = dtd_cov_1d_data2fit(data, btensors, cond_limit=1e-20, clip_eps=1e-20)
s0_convfit, d2_convfit, c4_convfit = convert_m(results)
V_MD2,V_iso2,V_shear2,MD,FA,V_MD,V_iso,V_MD1,V_iso1,V_shear,V_shear1,C_MD,C_mu,C_M,C_c,MKi,MKa,MKt,MKad,MK,MKd,uFA,S_I,S_A =  decode_m(
        d2_convfit, c4_convfit, reg=1e-20)

# fit the dot corrected data
results_dc = dtd_cov_1d_data2fit(data_dc, btensors, cond_limit=1e-20, clip_eps=1e-20)
s0_convfit_dc, d2_convfit_dc, c4_convfit_dc = convert_m(results_dc)
V_MD2_dc,V_iso2_dc,V_shear2_dc,MD_dc,FA_dc,V_MD_dc,V_iso_dc,V_MD1_dc,V_iso1_dc,V_shear_dc,V_shear1_dc,C_MD_dc,C_mu_dc,C_M_dc,C_c_dc,MKi_dc,MKa_dc,MKt_dc,MKad_dc,MK_dc,MKd_dc,uFA_dc,S_I_dc,S_A_dc =  decode_m(
        d2_convfit_dc, c4_convfit_dc, reg=1e-20)





fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(data, label='data')
ax1.plot(np.clip(_S_ens(btensors, s0_convfit, d2_convfit, c4_convfit), 0,1), label='predicted data')
ax1.set_ylabel('normalized signal $S/S_{0}$')
ax1.set_xlabel('acquisition number')
ax1.set_title('before dot-correction')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(data_dc, label='data')
ax2.plot(np.clip(_S_ens(btensors, s0_convfit_dc, d2_convfit_dc, c4_convfit_dc), 0,1), label='predicted data')
ax2.set_ylabel('normalized signal $S/S_{0}$')
ax2.set_xlabel('acquisition number')
ax2.set_title('after dot-correction')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

plt.suptitle('Comparison of signal and model-prediction for a wm voxel [40, 35, 40]')
plt.rcParams.update({'font.size': 18})
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.4,wspace=0.2)
plt.show()



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(data, label='data')
ax1.plot(np.clip(_S_ens(btensors, s0_convfit, d2_convfit, c4_convfit), 0,1), label='predicted data')
ax1.set_ylabel('signal')
ax1.set_xlabel('acquisition number')
ax1.set_title('before dot-correction')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(data_dc, label='data')
ax2.plot(np.clip(_S_ens(btensors, s0_convfit_dc, d2_convfit_dc, c4_convfit_dc), 0,1), label='predicted data')
ax2.set_ylabel('signal')
ax2.set_xlabel('acquisition number')
ax2.set_title('after dot-correction')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(np.abs(data - np.clip(_S_ens(btensors, s0_convfit, d2_convfit, c4_convfit), 0,1)), label='absolute error')
ax3.axhline(y = np.mean(np.abs(data - np.clip(_S_ens(btensors, s0_convfit, d2_convfit, c4_convfit), 0,1))), color='k', label='mean absolute error')
ax3.set_ylabel('absolute error')
ax3.set_xlabel('acquisition number')
ax3.set_ylim(0, 0.3)
ax3.legend(loc='upper right')

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(np.abs(data_dc - np.clip(_S_ens(btensors, s0_convfit_dc, d2_convfit_dc, c4_convfit_dc), 0,1)), label='absolute error')
ax4.axhline(y = np.mean(np.abs(data_dc - np.clip(_S_ens(btensors, s0_convfit_dc, d2_convfit_dc, c4_convfit_dc), 0,1))), color='k', label='mean absolute error')
ax4.set_ylabel('absolute error')
ax4.set_xlabel('acquisition number')
ax4.set_ylim(0, 0.3)
ax4.legend(loc='upper right')

plt.suptitle('Comparison of signal and model-prediction for a gm voxel')
plt.rcParams.update({'font.size': 14})
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.4,wspace=0.2)
plt.show()