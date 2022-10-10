import numpy as np
import os
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, readfile_btens, mean_Sb
from Definitions_smallscripts import monoexp, monoexp_fit, bi_exp, biexp_fit, cumulant_exp, cumexp_fit
import time


' Load data, btensors, bvalues and mask '
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')  # shape (90,60,130,331)

btensors = readfile_btens('btens_oneB0.txt')
bvals = np.loadtxt('bvals_oneB0.txt')  # bvals in s/mm2

btensors = btensors * 10 ** (-3)  # ms/ym^2
bvals = bvals * 10 ** (-3)

mask, affine = load_data('mask_pad.nii')


' first tray: monoexponential, biexponential and cumulant appraoch on single voxel of real data '
# choose a wm voxel
data = data_load[40,35,40,:]

# only fit dot-fraction on linear bvals
lin_idx = 6
b_mean, S_mean = mean_Sb(data, bvals)

bmean_lin = np.hstack([0, b_mean[lin_idx:]])
Smean_lin = np.hstack([S_mean[0], S_mean[lin_idx:]])

plt.plot(bmean_lin, Smean_lin)
plt.show()

# fit monoexp to initialize the cumulant
params_mono, pcov, init_mono = monoexp_fit(Smean_lin, bmean_lin)

# fit biexponential
params_biexp, pcov, init_biexp = biexp_fit(Smean_lin, bmean_lin, params_mono)

# fit cumulant
params_cumulant, pcov, init_cumulant = cumexp_fit(Smean_lin, bmean_lin, params_mono)
#params_cumulant_before_crash = (md_latest_init, mk_latest_init, df_latest_init)


plt.figure(figsize=(12, 7))
plt.rcParams.update({'font.size': 16})
plt.semilogy(bmean_lin, Smean_lin, label='mean lin data')
#plt.plot(bmean_lin, monoexp(bmean_lin, *init_mono), label='init mono')
plt.semilogy(bmean_lin, monoexp(bmean_lin, *params_mono), label='monoexponential approach')
plt.semilogy(bmean_lin, bi_exp(bmean_lin, *params_biexp), label='biexponential approach')
#plt.plot(bmean_lin, cumulant_exp(bmean_lin, *params_cumulant_before_crash), label='crash cumulant')
plt.semilogy(bmean_lin, cumulant_exp(bmean_lin, *params_cumulant), label='cumulant approach')
plt.xlabel('b-value [ms/$\mathrm{\mu m^2}$]', fontsize=16)
plt.ylabel('normalized signal S/S$_{0}$', fontsize=16)
plt.ylim(0.1, 1)
plt.grid(which='both', axis='both')
plt.title('Fitted diffusion models on a single white matter voxel at location [40,35,40]')
plt.legend(loc='upper right')
#plt.subplots_adjust(top=0.8,bottom=0.2,left=0.2,right=0.8,hspace=0.2,wspace=0.2)
plt.show()


' fit cumulant approach to a slice of the data '

slice_idx = 40

# choose a slice
data_slice = data_load[:, :, slice_idx, :]
mask_slice = mask[:, :, slice_idx]

#plt.imshow(data_slice[:, :, 1])
#plt.show()

lin_idx = 6

b_mean_tmp, S_mean_tmp = mean_Sb(data_slice[0,0], bvals)
b_mean_lin_data = np.hstack([0, b_mean_tmp[lin_idx:]])

S_mean_lin_data = np.zeros((data_slice.shape[0], data_slice.shape[1], b_mean_lin_data.shape[0]))
for xy in np.ndindex(mask_slice.shape):
    if mask_slice[xy]:
        S_mean_data = mean_Sb(data_slice[xy], bvals)[1]
        S_mean_lin_data[xy] = np.hstack([S_mean_data[0], S_mean_data[lin_idx:]])

start = time.time()
params_cumulant = np.zeros((data_slice.shape[0], data_slice.shape[1], 3))
for xy in np.ndindex(mask_slice.shape):
    if mask_slice[xy]:
        print(xy)

        # fit monoexp to initialize the cumulant
        params_mono, pcov, init_mono = monoexp_fit(S_mean_lin_data[xy], b_mean_lin_data)

        # fit cumulant
        params_cumulant[xy], pcov, init_cumulant = cumexp_fit(S_mean_lin_data[xy], b_mean_lin_data, params_mono)

print((time.time() - start)/60, 'minutes')


' fit biexponential to a slice of the data '

start = time.time()
params_biexp = np.zeros((data_slice.shape[0], data_slice.shape[1], 4))
for xy in np.ndindex(mask_slice.shape):
    if mask_slice[xy]:
        print(xy)

        # fit monoexp to initialize the cumulant
        params_mono, pcov, init_mono = monoexp_fit(S_mean_lin_data[xy], b_mean_lin_data)

        # fit cumulant
        params_biexp[xy], pcov, init_biexp = biexp_fit(S_mean_lin_data[xy], b_mean_lin_data, params_mono)
print((time.time() - start)/60, 'minutes')

#nib.Nifti1Image(params_biexp, affine).to_filename('dot_correction_biexp_params_10000its.nii')
#nib.Nifti1Image(params_cumulant, affine).to_filename('dot_correction_cumulant_params_100000its.nii')


'----------------------------------------------------------------------------------------------------'
# load data as a reference image
data_unnorm_load, affine = load_data('data_b0_pla_lin.nii')  # shape (90,60,130,331)
data_slice_b0 = data_unnorm_load[:, :, slice_idx, 0]


' Dot compartment correction '
# S_dot = (1-dotfrac) S_true + dotfrac
# S_true = S_corr = S_dot - dotfrac / (1-dotfrac)

epsilon = 1e-6

data_slice_cumulant_corrected_clipped = np.zeros((data_slice.shape))
data_slice_cumulant_corrected = np.zeros((data_slice.shape))
for xy in np.ndindex(mask_slice.shape):
    if mask_slice[xy]:
        data_slice_cumulant_corrected_clipped[xy] = np.clip((data_slice[xy] - params_cumulant[xy][2]) / (1 - params_cumulant[xy][2]), 0, 1)+epsilon
        data_slice_cumulant_corrected[xy] = (data_slice[xy] - params_cumulant[xy][2]) / (1 - params_cumulant[xy][2])
#nib.Nifti1Image(data_slice_cumulant_corrected_clipped, affine).to_filename('singleslice_data_cumulant_corrected_clipped.nii')

data_slice_biexp_corrected_clipped = np.zeros((data_slice.shape))
data_slice_biexp_corrected = np.zeros((data_slice.shape))
for xy in np.ndindex(mask_slice.shape):
    if mask_slice[xy]:
        data_slice_biexp_corrected_clipped[xy] = np.clip((data_slice[xy] - params_biexp[xy][3]) / (1 - params_biexp[xy][3]), 0, 1)+epsilon
        data_slice_biexp_corrected[xy] = (data_slice[xy] - params_biexp[xy][3]) / (1 - params_biexp[xy][3])
#nib.Nifti1Image(data_slice_biexp_corrected_clipped, affine).to_filename('singleslice_data_biexp_corrected_clipped.nii')


#params_cumulant, affine = load_data('dot_correction_cumulant_params_100000its.nii')

fig = plt.figure()
im_ratio = data_slice_b0.shape[0]/data_slice_b0.shape[1]

ax1 = fig.add_subplot(2,3,1)
im1 = ax1.imshow(data_slice_b0, cmap='gray')
ax1.set_title('unnormalized b0 data')
plt.colorbar(im1, fraction=0.047*im_ratio)

ax2 = fig.add_subplot(2,3,2)
im2 = ax2.imshow(params_cumulant[:, :, 2], cmap='gray')
ax2.set_title('cumulant\n estimated dot-fraction')
plt.colorbar(im2, fraction=0.047*im_ratio)

ax3 = fig.add_subplot(2,3,3)
im3 = ax3.imshow(data_slice_cumulant_corrected_clipped[:, :, 1], cmap='gray')#,vmin=-1, vmax=1)
ax3.set_title('cumulant\n normalized corrected data (clipped: 0,1)')
plt.colorbar(im3, fraction=0.047*im_ratio)

ax4 = fig.add_subplot(2,3,5)
im4 = ax4.imshow(params_biexp[:, :, 3], cmap='gray')#,vmin=-1, vmax=1)
ax4.set_title('biexp\n estimated dot-fraction')
plt.colorbar(im4, fraction=0.047*im_ratio)

ax5 = fig.add_subplot(2,3,6)
im5 = ax5.imshow(data_slice_biexp_corrected_clipped[:, :, 1], cmap='gray')#,vmin=-1, vmax=1)
ax5.set_title('biexp\n normalized corrected data (clipped: 0,1)')
plt.colorbar(im5, fraction=0.047*im_ratio)

plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.2,wspace=0.3)
plt.rcParams.update({'font.size': 12})
plt.show()

plt.suptitle('Dot-correction via biexp. and cumulant fit')
plt.plot()


# look at a wm voxel
plt.plot(data_slice[35,40,:], label='uncorrected data (normalized)')
plt.plot(data_slice_biexp_corrected_clipped[35,40,:], label='biexp corrected data (clipped)')
#plt.plot(data_slice_cumulant_corrected_clipped[35,40,:], label='cumulant corrected data (clipped)')
plt.legend()
plt.show()

plt.plot(data_slice[35,40,:], label='uncorrected data (normalized)')
plt.plot(data_slice_biexp_corrected[35,40,:], label='biexp corrected data (unclipped)')
#plt.plot(data_slice_cumulant_corrected[35,40,:], label='cumulant corrected data (unclipped)')
plt.legend()
plt.show()




' another plot '

params_biexp, affine = load_data('dot_correction_biexp_params_10000its_fulldata.nii')
data_biexp_dotcorrected, affine = load_data('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')

fig = plt.figure()
im_ratio = data_slice_b0.shape[0]/data_slice_b0.shape[1]
plt.rcParams.update({'font.size': 12})

ax1 = fig.add_subplot(1,3,1)
im1 = ax1.imshow(data_slice_b0, cmap='gray')
ax1.set_title('unnormalized b0 data')
plt.colorbar(im1, fraction=0.047*im_ratio)

ax2 = fig.add_subplot(1, 3, 2)
im2 = ax2.imshow(params_biexp[:, :, slice_idx, 3], cmap='gray')
ax2.set_title('biexponential approach\n estimated dot-fraction')
plt.colorbar(im2, fraction=0.047 * im_ratio)

ax3 = fig.add_subplot(1, 3, 3)
im3 = ax3.imshow(data_biexp_dotcorrected[:, :, slice_idx, 1], cmap='gray')  # ,vmin=-1, vmax=1)
ax3.set_title('biexponential approach\n normalized corrected data (clipped: 0,1)')
plt.colorbar(im3, fraction=0.047 * im_ratio)

plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.2,wspace=0.2)
plt.show()

