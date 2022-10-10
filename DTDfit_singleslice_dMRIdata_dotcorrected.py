import numpy as np
import os
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, readfile_btens, mean_bvals, mean_signal, mean_Sb
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit, dtd_cov_1d_data2fit, convert_m
from mpl_toolkits.axes_grid1 import make_axes_locatable

slice_idx = 40

' Load data, btensors, bvalues and mask '
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')  # shape (90,60,130,331)
data_slice_corrected = data_load[:,:,slice_idx,:301]

#data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')  # shape (90,60,130,331)
#data_slice_corrected = data_load[:, :, slice_idx, :301] # do the fit without the last linear shell

btensors = readfile_btens('btens_oneB0.txt')
bvals = np.loadtxt('bvals_oneB0.txt')  # bvals in s/mm2

btensors = btensors * 10 ** (-3)  # ms/ym^2
bvals = bvals * 10 ** (-3)

mask, affine = load_data('mask_pad.nii')
mask_slice = mask[:, :, slice_idx]

' fit the DTD model '
K = 28 # number of variables that the fitfunktion has as uotput (from linear least squares fit)
results = np.zeros(data_slice_corrected.shape[:2] + (K,))
print('Fit results')

for xy in np.ndindex(mask_slice.shape):  # loop in N-dimension, xyz is a tuple (x,y,z)
    if mask_slice[xy]:  # if in mask
        #results[xyz] = dtd_cov_1d_data2fit_v1(data[xyz], btensors, cond_limit=1e-10, clip_eps=1e-16) # fit
        results[xy] = dtd_cov_1d_data2fit(data_slice_corrected[xy], btensors, cond_limit=1e-20, clip_eps=1e-20)


# now: fit all of the values
V_MD2_fit = np.zeros(data_slice_corrected.shape[:2])
V_iso2_fit = np.zeros(data_slice_corrected.shape[:2])
V_shear2_fit = np.zeros(data_slice_corrected.shape[:2])
MD_fit = np.zeros(data_slice_corrected.shape[:2])
FA_fit = np.zeros(data_slice_corrected.shape[:2])
V_MD_fit = np.zeros(data_slice_corrected.shape[:2])
V_iso_fit = np.zeros(data_slice_corrected.shape[:2])
V_MD1_fit = np.zeros(data_slice_corrected.shape[:2])
V_iso1_fit = np.zeros(data_slice_corrected.shape[:2])
V_shear_fit = np.zeros(data_slice_corrected.shape[:2])
V_shear1_fit = np.zeros(data_slice_corrected.shape[:2])
C_MD_fit = np.zeros(data_slice_corrected.shape[:2])
C_mu_fit = np.zeros(data_slice_corrected.shape[:2])
C_M_fit = np.zeros(data_slice_corrected.shape[:2])
C_c_fit = np.zeros(data_slice_corrected.shape[:2])
MKi_fit = np.zeros(data_slice_corrected.shape[:2])
MKa_fit = np.zeros(data_slice_corrected.shape[:2])
MKt_fit = np.zeros(data_slice_corrected.shape[:2])
MKad_fit = np.zeros(data_slice_corrected.shape[:2])
MK_fit = np.zeros(data_slice_corrected.shape[:2])
MKd_fit = np.zeros(data_slice_corrected.shape[:2])
uFA_fit = np.zeros(data_slice_corrected.shape[:2])
S_I_fit = np.zeros(data_slice_corrected.shape[:2])
S_A_fit = np.zeros(data_slice_corrected.shape[:2])

print('calculate parameters')

for xy in np.ndindex(results.shape[:2]):
    # get the ordered solution of the fit
    s0_convfit, d2_convfit, c4_convfit = convert_m(results[xy])
    # get any other parameters from the fit
    V_MD2_fit[xy], V_iso2_fit[xy], V_shear2_fit[xy], MD_fit[xy], FA_fit[xy], V_MD_fit[xy], V_iso_fit[xy], V_MD1_fit[xy], V_iso1_fit[xy], V_shear_fit[xy], V_shear1_fit[xy], C_MD_fit[xy], \
    C_mu_fit[xy], C_M_fit[xy], C_c_fit[xy], MKi_fit[xy], MKa_fit[xy], MKt_fit[xy], MKad_fit[xy], MK_fit[xy], MKd_fit[xy], uFA_fit[xy], S_I_fit[xy], S_A_fit[xy] = decode_m(
        d2_convfit, c4_convfit, reg=1e-20)


' --- plot ---'
# load data as a reference image
data_unnorm_load, affine = load_data('data_b0_pla_lin.nii')  # shape (90,60,130,331)
data_slice_b0 = data_unnorm_load[:, :, slice_idx, 0]


def subplot_conf(img_data, pos=(111), title='string', a=0, b=1):
    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    ax1 = fig.add_subplot(pos)
    im1 = ax1.imshow(img_data, cmap='gray', vmin=a, vmax=b)
    ax1.set_title(title)
    ax1.axis('off')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', format='%.0e')
    #fig.colorbar(im1, cax=cax, orientation='vertical', format= ticker.FuncFormatter(fmt))


fig = plt.figure(figsize=(16, 12))

subplot_conf(data_slice_b0, pos=(241), title='non-dw signal')
subplot_conf(MD_fit, pos=(242), title='MD')
subplot_conf(FA_fit, pos=(243), title='FA')
subplot_conf(C_M_fit, pos=(244), title='$C_M = FA²$')
subplot_conf(C_c_fit, pos=(245), title='$C_c$')
subplot_conf(C_MD_fit, pos=(246), title='$C_{MD}$')
subplot_conf(uFA_fit, pos=(247), title='$\mu FA$')
subplot_conf(C_mu_fit, pos=(248), title='$C_\mu = \mu FA²$')
fig.tight_layout()
plt.show()
