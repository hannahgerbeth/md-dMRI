import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from Definitions import DT_orientation, DT_evals, DT_evecs, FA_gen, MD_gen, Diffusion_Tensors_manual, S_dis, noisy_signal
from Definitions import dtd_cov_1d_data2fit_v1, plot_tensors, S_cum_ens
from Definitions_smallscripts import load_data, readfile_btens
from dtd_cov_MPaquette import convert_m, decode_m
import time
import os

start_time = time.time()
"---- Header Start ----"

np.random.seed(0)

def run_sim(mask, signal_b0data, md_map, fa_map, sigma_map, btens, SNR, D_shape='lin', N=10000, k=100, threshold=0.8):
    mu = [1., 0, 0]
    # Output:
    # 0 V_MD2_fit
    # 1 V_iso2_fit
    # 2 V_shear2_fit
    # 3 MD_fit
    # 4 FA_fit
    # 5 V_MD_fit
    # 6 V_iso_fit
    # 7 V_MD1_fit
    # 8 V_iso1_fit
    # 9 V_shear_fit
    # 10 V_shear1_fit
    # 11 C_MD_fit
    # 12 C_mu_fit
    # 13 C_M_fit
    # 14 C_c_fit
    # 15 MKi_fit
    # 16 MKa_fit
    # 17 MKt_fit
    # 18 MKad_fit
    # 19 MK_fit
    # 20 MKd_fit
    # 21 uFA_fit
    # 22 S_I_fit
    # 23 S_A_fit

    noise_sigma_mask = sigma_map * mask
    md_mask = md_map * mask
    fa_mask = fa_map * mask

    md_gt = np.mean(md_mask[np.where(md_mask!= 0)])
    md_gt_std = np.std(md_mask[np.where(md_mask != 0)])
    md_dis = MD_gen(N, md_gt, md_gt_std)

    fa_gt = np.mean(fa_map[np.where(mask != 0)])
    fa_gt_std = np.std(fa_map[np.where(mask != 0)])

    #fa_gt = np.mean(fa_mask[np.where(fa_mask!=0)])
    #fa_gt_std = np.std(fa_mask[np.where(fa_mask!=0)])
    fa_dist = FA_gen(N, fa_gt, fa_gt_std)

    # generate synthetic DTD for tissue
    dt_orien = DT_orientation(N, k, mu, threshold)
    dt_evecs = DT_evecs(N, dt_orien)
    dt_evals = DT_evals(D_shape, md_dis, fa_dist)
    dtens = Diffusion_Tensors_manual(dt_evecs, dt_evals)


    # generate normalized signal wihtout noise
    signal_normalized = S_dis(btens, dtens)

    sig_b0_unnormalized = signal_b0data * mask
    S0 = np.mean(sig_b0_unnormalized[np.where(sig_b0_unnormalized != 0)])
    signal_unnormalized = signal_normalized * S0
    #print('S0', S0)

    # convert SNR and sigma
    # snr = signal/sigma
    #sigma_mean = np.mean(noise_sigma_mask[np.where(noise_sigma_mask != 0)])
    #SNR = S0 / np.mean(noise_sigma_mask[np.where(noise_sigma_mask != 0)])
    sigma_for_noise = S0/SNR
    #print('sigma for noise', sigma_for_noise)

    signal_unnormalized_noisy = noisy_signal(signal_unnormalized, sigma_for_noise)[0]
    print("--- %s seconds ---" % (time.time() - start_time))

    fit = dtd_cov_1d_data2fit_v1(signal_unnormalized_noisy, btensors)
    s0_convfit, d2_convfit, c4_convfit = convert_m(fit)

    # get any other parameters from the fit
    V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit = decode_m(
        d2_convfit, c4_convfit, reg=1e-4)
    print("--- %s seconds ---" % (time.time() - start_time))

    return V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit

def run_sim_v2(S0, btens, dtens, SNR):
    # Output:
    # 0 V_MD2_fit
    # 1 V_iso2_fit
    # 2 V_shear2_fit
    # 3 MD_fit
    # 4 FA_fit
    # 5 V_MD_fit
    # 6 V_iso_fit
    # 7 V_MD1_fit
    # 8 V_iso1_fit
    # 9 V_shear_fit
    # 10 V_shear1_fit
    # 11 C_MD_fit
    # 12 C_mu_fit
    # 13 C_M_fit
    # 14 C_c_fit
    # 15 MKi_fit
    # 16 MKa_fit
    # 17 MKt_fit
    # 18 MKad_fit
    # 19 MK_fit
    # 20 MKd_fit
    # 21 uFA_fit
    # 22 S_I_fit
    # 23 S_A_fit

    #print("--- time for dtd %s seconds ---" % (time.time() - start_time))

    # generate normalized signal wihtout noise
    #signal_normalized = S_dis(btens, dtens)
    signal_normalized = S_cum_ens(btens, dtens)
    signal_unnormalized = signal_normalized * S0
    #print('S0', S0)

    # convert SNR and sigma
    # snr = signal/sigma
    sigma_for_noise = S0/SNR

    signal_unnormalized_noisy = noisy_signal(signal_unnormalized, sigma_for_noise)[0]
    #print("--- time for signal gen %s seconds ---" % (time.time() - start_time))

    fit = dtd_cov_1d_data2fit_v1(signal_unnormalized_noisy, btensors)
    s0_convfit, d2_convfit, c4_convfit = convert_m(fit)

    # get any other parameters from the fit
    V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit = decode_m(
        d2_convfit, c4_convfit, reg=1e-10)
    #print("--- time for fit %s seconds ---" % (time.time() - start_time))

    return V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit

def run_sim_noiseless(S0, btens, dtens):
    # Output:
    # 0 V_MD2_fit
    # 1 V_iso2_fit
    # 2 V_shear2_fit
    # 3 MD_fit
    # 4 FA_fit
    # 5 V_MD_fit
    # 6 V_iso_fit
    # 7 V_MD1_fit
    # 8 V_iso1_fit
    # 9 V_shear_fit
    # 10 V_shear1_fit
    # 11 C_MD_fit
    # 12 C_mu_fit
    # 13 C_M_fit
    # 14 C_c_fit
    # 15 MKi_fit
    # 16 MKa_fit
    # 17 MKt_fit
    # 18 MKad_fit
    # 19 MK_fit
    # 20 MKd_fit
    # 21 uFA_fit
    # 22 S_I_fit
    # 23 S_A_fit

    #print("--- time for dtd %s seconds ---" % (time.time() - start_time))

    # generate normalized signal wihtout noise
    #signal_normalized = S_dis(btens, dtens)
    signal_normalized = S_cum_ens(btens, dtens)
    signal_unnormalized = signal_normalized * S0
    #print('S0', S0)

    # convert SNR and sigma
    # snr = signal/sigma
    #sigma_for_noise = S0/SNR

    #signal_unnormalized_noisy = noisy_signal(signal_unnormalized, sigma_for_noise)[0]
    #print("--- time for signal gen %s seconds ---" % (time.time() - start_time))

    fit = dtd_cov_1d_data2fit_v1(signal_unnormalized, btensors)
    s0_convfit, d2_convfit, c4_convfit = convert_m(fit)

    # get any other parameters from the fit
    V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit = decode_m(
        d2_convfit, c4_convfit, reg=1e-10)
    #print("--- time for fit %s seconds ---" % (time.time() - start_time))

    return V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit


'------------------------------------------------------------------------------------------------------------'
' load data '

data_signal, affine = load_data('data_concatenate_lin_pla.nii')
data_signal_b0 = data_signal[:, :, :, 0]
gm_mask, affine = load_data('GMmask.nii')
wm_mask, affine = load_data('WMmask.nii')
sigma_map, affine = load_data('noise_sigmas_reshape_pad.nii')

# read an txt-file and bring the values into an array-format
filename = str('btens_oneB0.txt')
btensors = readfile_btens(filename) # output: ndarray (n, )
btensors = btensors * 10**(-3)


# go to the directory of fitted diffusion maps
# load MD-map, FA-map and micro-FA-map
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/211021_Heschl_Bruker_Magdeburg/simulation_debug")

md_map, affine = load_data('MD_fit.nii')
fa_map, affine = load_data('FA_fit.nii')
ufa_map, affine = load_data('uFA_fit.nii')

'-----------------------------------------------------------------------------'
'prepare simulation for white matter'
noise_sigma_mask = sigma_map * wm_mask

md_gt_white = np.mean(md_map[np.where(wm_mask == 1)])
print('white matter mean MD from data', md_gt_white)
md_gt_std_white = np.std(md_map[np.where(wm_mask==1)])

fa_gt_white = np.mean(fa_map[np.where(wm_mask ==1)])
print('white matter mean FA from data', fa_gt_white)
fa_gt_std_white = np.std(fa_map[np.where(wm_mask ==1)])

ufa_gt_white = np.mean(ufa_map[np.where(wm_mask ==1)])
print('white matter mean uFA from data', ufa_gt_white)
ufa_gt_std_white = np.std(ufa_map[np.where(wm_mask ==1)])


'-------------------- first configuration with k = 50 -------------------------------'
N = 1000
k = 50
mu = [1., 0., 0.]
threshold=0.8
D_shape='lin'

md_dis_white = MD_gen(N, md_gt_white, md_gt_std_white)
fa_dist_white = FA_gen(N, fa_gt_white, fa_gt_std_white)

# generate synthetic DTD for tissue
dt_orien = DT_orientation(N, k, mu, threshold)
dt_evecs = DT_evecs(N, dt_orien)
dt_evals = DT_evals(D_shape, md_dis_white, fa_dist_white)
dtens_white_50 = Diffusion_Tensors_manual(dt_evecs, dt_evals)

sig_b0_unnormalized_white = data_signal[:, :, :, 0] * wm_mask
S0_white_50 = np.mean(sig_b0_unnormalized_white[np.where(sig_b0_unnormalized_white != 0)])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_tensors(dtens_white_50[0:100], fig, ax, factor=2)
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plt.title('White matter DTD, k={}'.format(k), fontsize=30)
plt.show()


'run simulation for wm without snr'
Number_runs = 100

V_MD2_fit = np.zeros((Number_runs))
V_iso2_fit = np.zeros((Number_runs))
V_shear2_fit = np.zeros(( Number_runs))
MD_fit = np.zeros((Number_runs))
FA_fit = np.zeros((Number_runs))
V_MD_fit = np.zeros(( Number_runs))
V_iso_fit = np.zeros(( Number_runs))
V_MD1_fit = np.zeros(( Number_runs))
V_iso1_fit = np.zeros((Number_runs))
V_shear_fit  = np.zeros((Number_runs))
V_shear1_fit = np.zeros((Number_runs))
C_MD_fit = np.zeros((Number_runs))
C_mu_fit = np.zeros((Number_runs))
C_M_fit = np.zeros((Number_runs))
C_c_fit = np.zeros(( Number_runs))
MKi_fit = np.zeros((Number_runs))
MKa_fit = np.zeros(( Number_runs))
MKt_fit = np.zeros((Number_runs))
MKad_fit = np.zeros((Number_runs))
MK_fit = np.zeros(( Number_runs))
MKd_fit = np.zeros(( Number_runs))
uFA_fit = np.zeros((Number_runs))
S_I_fit = np.zeros(( Number_runs))
S_A_fit = np.zeros((Number_runs))


for i in range(Number_runs):
    #print(i)
    V_MD2_fit[i], V_iso2_fit[i], V_shear2_fit[i], MD_fit[i], FA_fit[i], V_MD_fit[i], \
    V_iso_fit[i], V_MD1_fit[i], V_iso1_fit[i], V_shear_fit[i], V_shear1_fit[i], C_MD_fit[i], \
    C_mu_fit[i], C_M_fit[i], C_c_fit[i], MKi_fit[i], MKa_fit[i], MKt_fit[i], MKad_fit[i], \
    MK_fit[i], MKd_fit[i], uFA_fit[i], S_I_fit[i], S_A_fit[i] = run_sim_noiseless(S0_white_50, btensors, dtens_white_50)


print('--------------------------------------------------------')
print('White matter MD noiseless simulation:', np.mean(MD_fit), 'p/m', np.std(MD_fit))
print('White matter FA noiseless simulation:', np.mean(FA_fit), 'p/m', np.std(FA_fit))
print('White matter uFA noiseless simulation:', np.mean(uFA_fit), 'p/m', np.std(uFA_fit))
print('--------------------------------------------------------')

MD_fit_noiseless_white_50 = np.mean(MD_fit)
MD_fit_noiseless_std_white_50 = np.std(MD_fit)

FA_fit_noiseless_white_50 = np.mean(FA_fit)
FA_fit_noiseless_std_white_50 = np.std(FA_fit)

uFA_fit_noiseless_white_50 = np.mean(uFA_fit)
uFA_fit_noiseless_std_white_50 = np.std(uFA_fit)


' run simulation for white matter with SNR'

Number_runs = 100
# SNR in pre-scan data: SNR = 295.74
SNR = np.arange(50, 320, 10)

V_MD2_fit = np.zeros((len(SNR), Number_runs))
V_iso2_fit = np.zeros((len(SNR), Number_runs))
V_shear2_fit = np.zeros((len(SNR), Number_runs))
MD_fit = np.zeros((len(SNR), Number_runs))
FA_fit = np.zeros((len(SNR), Number_runs))
V_MD_fit = np.zeros((len(SNR), Number_runs))
V_iso_fit = np.zeros((len(SNR), Number_runs))
V_MD1_fit = np.zeros((len(SNR), Number_runs))
V_iso1_fit = np.zeros((len(SNR), Number_runs))
V_shear_fit  = np.zeros((len(SNR), Number_runs))
V_shear1_fit = np.zeros((len(SNR), Number_runs))
C_MD_fit = np.zeros((len(SNR), Number_runs))
C_mu_fit = np.zeros((len(SNR), Number_runs))
C_M_fit = np.zeros((len(SNR), Number_runs))
C_c_fit = np.zeros((len(SNR), Number_runs))
MKi_fit = np.zeros((len(SNR), Number_runs))
MKa_fit = np.zeros((len(SNR), Number_runs))
MKt_fit = np.zeros((len(SNR), Number_runs))
MKad_fit = np.zeros((len(SNR), Number_runs))
MK_fit = np.zeros((len(SNR), Number_runs))
MKd_fit = np.zeros((len(SNR), Number_runs))
uFA_fit = np.zeros((len(SNR), Number_runs))
S_I_fit = np.zeros((len(SNR), Number_runs))
S_A_fit = np.zeros((len(SNR), Number_runs))


for i in range(Number_runs):
    print(i)
    for j in range(len(SNR)):
        print('SNR', SNR[j])
        #V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        #V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        #C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        #MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim(wm_mask, data_signal_b0, md_map, fa_map, sigma_map, btensors, SNR[j])
        V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim_v2(S0_white_50, btensors, dtens_white_50, SNR[j])
    print("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(figsize=(13, 8))
plt.rcParams.update({'font.size': 18})

plt.errorbar(SNR, np.mean(MD_fit, axis=1), yerr=np.std(MD_fit, axis=1),fmt='o')
plt.axhline(y=md_gt_white, xmin=0, xmax=np.max(SNR),color='r', label='mean MD from fit = {} $\mu$m²/ms'.format(np.round(md_gt_white, decimals=3)))
plt.axhline(y=MD_fit_noiseless_white_50, xmin=0, xmax=np.max(SNR), color='k',label='mean MD from noiseless simulation = {} $\mu$m²/ms'.format(np.round(MD_fit_noiseless_white_50, decimals=2)))
plt.xlabel('SNR')
plt.ylabel('MD [$\mu$m²/ms]')
plt.title('MD(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()

plt.figure(figsize=(13, 8))
plt.rcParams.update({'font.size': 18})
plt.errorbar(SNR, np.mean(FA_fit, axis=1), yerr=np.std(FA_fit, axis=1),fmt='o')
plt.axhline(y=fa_gt_white, xmin=0, xmax=np.max(SNR),color='r', label='mean FA from fit = {} '.format(np.round(fa_gt_white, decimals=3)))
plt.axhline(y=FA_fit_noiseless_white_50, xmin=0, xmax=np.max(SNR),color='k', label='mean FA from noiseless simulation = {} '.format(np.round(FA_fit_noiseless_white_50, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('FA')
plt.title('FA(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(13, 8))
plt.rcParams.update({'font.size': 18})
plt.errorbar(SNR, np.mean(uFA_fit, axis=1), yerr=np.std(uFA_fit, axis=1),fmt='o')
plt.axhline(y=fa_gt_white, xmin=0, xmax=np.max(SNR),color='r', label='mean uFA from fit = {} '.format(np.round(fa_gt_white, decimals=3)))
plt.axhline(y=uFA_fit_noiseless_white_50, xmin=0, xmax=np.max(SNR), color='k',label='mean uFA from noiseless simulation = {} '.format(np.round(uFA_fit_noiseless_white_50, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('$\mu$FA')
plt.title('$\mu$FA(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.show()


#nib.Nifti1Image(MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format(MD_fit, Number_runs))
#nib.Nifti1Image(FA_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format(FA_fit, Number_runs))

"""
# save back nifti of result with same affine as "output_filename"
nib.Nifti1Image(V_MD2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD2_fit', Number_runs))
nib.Nifti1Image(V_iso2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso2_fit', Number_runs))
nib.Nifti1Image(V_shear2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear2_fit', Number_runs))
nib.Nifti1Image(MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MD_fit', Number_runs))
nib.Nifti1Image(FA_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('FA_fit', Number_runs))
nib.Nifti1Image(V_MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD_fit', Number_runs))
nib.Nifti1Image(V_iso_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso_fit', Number_runs))
nib.Nifti1Image(V_MD1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD1_fit', Number_runs))
nib.Nifti1Image(V_iso1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso1_fit', Number_runs))
nib.Nifti1Image(V_shear_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear_fit', Number_runs))
nib.Nifti1Image(V_shear1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear1_fit', Number_runs))
nib.Nifti1Image(C_MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_MD_fit', Number_runs))
nib.Nifti1Image(C_mu_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_mu_fit', Number_runs))
nib.Nifti1Image(C_M_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_M_fit', Number_runs))
nib.Nifti1Image(C_c_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_c_fit', Number_runs))
nib.Nifti1Image(MKi_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKi_fit', Number_runs))
nib.Nifti1Image(MKa_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKa_fit', Number_runs))
nib.Nifti1Image(MKt_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKt_fit', Number_runs))
nib.Nifti1Image(MKad_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKad_fit', Number_runs))
nib.Nifti1Image(MK_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MK_fit', Number_runs))
nib.Nifti1Image(MKd_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKd_fit', Number_runs))
nib.Nifti1Image(uFA_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('uFA_fit', Number_runs))
nib.Nifti1Image(S_I_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('S_I_fit', Number_runs))
nib.Nifti1Image(S_A_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('S_A_fit', Number_runs))
"""

'-------------------- first configuration with k = 100 -------------------------------'
N = 1000
k = 100

# generate synthetic DTD for tissue
dt_orien = DT_orientation(N, k, mu, threshold)
dt_evecs = DT_evecs(N, dt_orien)
dt_evals = DT_evals(D_shape, md_dis_white, fa_dist_white)
dtens_white_100 = Diffusion_Tensors_manual(dt_evecs, dt_evals)

sig_b0_unnormalized_white_100 = data_signal[:, :, :, 0] * wm_mask
S0_white_100= np.mean(sig_b0_unnormalized_white_100[np.where(sig_b0_unnormalized_white_100 != 0)])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_tensors(dtens_white_100[0:100], fig, ax, factor=2)
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plt.title('White matter DTD, k={}'.format(k), fontsize=30)
plt.show()


'run simulation for wm without snr'
Number_runs = 100

V_MD2_fit = np.zeros((Number_runs))
V_iso2_fit = np.zeros((Number_runs))
V_shear2_fit = np.zeros(( Number_runs))
MD_fit = np.zeros((Number_runs))
FA_fit = np.zeros((Number_runs))
V_MD_fit = np.zeros(( Number_runs))
V_iso_fit = np.zeros(( Number_runs))
V_MD1_fit = np.zeros(( Number_runs))
V_iso1_fit = np.zeros((Number_runs))
V_shear_fit  = np.zeros((Number_runs))
V_shear1_fit = np.zeros((Number_runs))
C_MD_fit = np.zeros((Number_runs))
C_mu_fit = np.zeros((Number_runs))
C_M_fit = np.zeros((Number_runs))
C_c_fit = np.zeros(( Number_runs))
MKi_fit = np.zeros((Number_runs))
MKa_fit = np.zeros(( Number_runs))
MKt_fit = np.zeros((Number_runs))
MKad_fit = np.zeros((Number_runs))
MK_fit = np.zeros(( Number_runs))
MKd_fit = np.zeros(( Number_runs))
uFA_fit = np.zeros((Number_runs))
S_I_fit = np.zeros(( Number_runs))
S_A_fit = np.zeros((Number_runs))


for i in range(Number_runs):
    #print(i)
    V_MD2_fit[i], V_iso2_fit[i], V_shear2_fit[i], MD_fit[i], FA_fit[i], V_MD_fit[i], \
    V_iso_fit[i], V_MD1_fit[i], V_iso1_fit[i], V_shear_fit[i], V_shear1_fit[i], C_MD_fit[i], \
    C_mu_fit[i], C_M_fit[i], C_c_fit[i], MKi_fit[i], MKa_fit[i], MKt_fit[i], MKad_fit[i], \
    MK_fit[i], MKd_fit[i], uFA_fit[i], S_I_fit[i], S_A_fit[i] = run_sim_noiseless(S0_white_100, btensors, dtens_white_100)


print('--------------------------------------------------------')
print('White matter MD noiseless simulation:', np.mean(MD_fit), 'p/m', np.std(MD_fit))
print('White matter FA noiseless simulation:', np.mean(FA_fit), 'p/m', np.std(FA_fit))
print('White matter uFA noiseless simulation:', np.mean(uFA_fit), 'p/m', np.std(uFA_fit))
print('--------------------------------------------------------')

MD_fit_noiseless_white_100 = np.mean(MD_fit)
MD_fit_noiseless_std_white_100 = np.std(MD_fit)

FA_fit_noiseless_white_100 = np.mean(FA_fit)
FA_fit_noiseless_std_white_100 = np.std(FA_fit)

uFA_fit_noiseless_white_100 = np.mean(uFA_fit)
uFA_fit_noiseless_std_white_100 = np.std(uFA_fit)


' run simulation for white matter with SNR'

Number_runs = 100
#SNR = 295.74
SNR = np.arange(50, 320, 10)

V_MD2_fit = np.zeros((len(SNR), Number_runs))
V_iso2_fit = np.zeros((len(SNR), Number_runs))
V_shear2_fit = np.zeros((len(SNR), Number_runs))
MD_fit = np.zeros((len(SNR), Number_runs))
FA_fit = np.zeros((len(SNR), Number_runs))
V_MD_fit = np.zeros((len(SNR), Number_runs))
V_iso_fit = np.zeros((len(SNR), Number_runs))
V_MD1_fit = np.zeros((len(SNR), Number_runs))
V_iso1_fit = np.zeros((len(SNR), Number_runs))
V_shear_fit  = np.zeros((len(SNR), Number_runs))
V_shear1_fit = np.zeros((len(SNR), Number_runs))
C_MD_fit = np.zeros((len(SNR), Number_runs))
C_mu_fit = np.zeros((len(SNR), Number_runs))
C_M_fit = np.zeros((len(SNR), Number_runs))
C_c_fit = np.zeros((len(SNR), Number_runs))
MKi_fit = np.zeros((len(SNR), Number_runs))
MKa_fit = np.zeros((len(SNR), Number_runs))
MKt_fit = np.zeros((len(SNR), Number_runs))
MKad_fit = np.zeros((len(SNR), Number_runs))
MK_fit = np.zeros((len(SNR), Number_runs))
MKd_fit = np.zeros((len(SNR), Number_runs))
uFA_fit = np.zeros((len(SNR), Number_runs))
S_I_fit = np.zeros((len(SNR), Number_runs))
S_A_fit = np.zeros((len(SNR), Number_runs))


for i in range(Number_runs):
    print(i)
    for j in range(len(SNR)):
        print('SNR', SNR[j])
        #V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        #V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        #C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        #MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim(wm_mask, data_signal_b0, md_map, fa_map, sigma_map, btensors, SNR[j])
        V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim_v2(S0_white_100, btensors, dtens_white_100, SNR[j])
    print("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(MD_fit, axis=1), yerr=np.std(MD_fit, axis=1),fmt='o')
plt.axhline(y=md_gt_white, xmin=0, xmax=np.max(SNR), color='r',label='mean MD from fit = {} $\mu$m²/ms'.format(np.round(md_gt_white, decimals=3)))
plt.axhline(y=MD_fit_noiseless_white_100, xmin=0, xmax=np.max(SNR), color='k',label='mean MD from noiseless simulation = {} $\mu$m²/ms'.format(np.round(MD_fit_noiseless_white_100, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('MD [$\mu$m²/ms]')
plt.title('MD(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper left')
plt.rcParams.update({'font.size': 18})
plt.show()

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(FA_fit, axis=1), yerr=np.std(FA_fit, axis=1),fmt='o')
plt.axhline(y=fa_gt_white, xmin=0, xmax=np.max(SNR), color='r',label='mean FA from fit = {}'.format(np.round(fa_gt_white, decimals=3)))
plt.axhline(y=FA_fit_noiseless_white_100, xmin=0, xmax=np.max(SNR), color='k',label='mean FA from noiseless simulation = {}'.format(np.round(FA_fit_noiseless_white_100, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('FA')
plt.title('FA(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(uFA_fit, axis=1), yerr=np.std(uFA_fit, axis=1),fmt='o')
plt.axhline(y=fa_gt_white, xmin=0, xmax=np.max(SNR), color='r',label='mean uFA from fit = {}'.format(np.round(fa_gt_white, decimals=3)))
plt.axhline(y=uFA_fit_noiseless_white_100, xmin=0, xmax=np.max(SNR),color='k', label='mean uFA from noiseless simulation = {}'.format(np.round(uFA_fit_noiseless_white_100, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('$\mu$FA')
plt.title('$\mu$FA(SNR) for a synthetic DTD of WM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()


"""
# save back nifti of result with same affine as "output_filename"
nib.Nifti1Image(V_MD2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD2_fit', Number_runs))
nib.Nifti1Image(V_iso2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso2_fit', Number_runs))
nib.Nifti1Image(V_shear2_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear2_fit', Number_runs))
nib.Nifti1Image(MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MD_fit', Number_runs))
nib.Nifti1Image(FA_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('FA_fit', Number_runs))
nib.Nifti1Image(V_MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD_fit', Number_runs))
nib.Nifti1Image(V_iso_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso_fit', Number_runs))
nib.Nifti1Image(V_MD1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_MD1_fit', Number_runs))
nib.Nifti1Image(V_iso1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_iso1_fit', Number_runs))
nib.Nifti1Image(V_shear_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear_fit', Number_runs))
nib.Nifti1Image(V_shear1_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('V_shear1_fit', Number_runs))
nib.Nifti1Image(C_MD_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_MD_fit', Number_runs))
nib.Nifti1Image(C_mu_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_mu_fit', Number_runs))
nib.Nifti1Image(C_M_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_M_fit', Number_runs))
nib.Nifti1Image(C_c_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('C_c_fit', Number_runs))
nib.Nifti1Image(MKi_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKi_fit', Number_runs))
nib.Nifti1Image(MKa_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKa_fit', Number_runs))
nib.Nifti1Image(MKt_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKt_fit', Number_runs))
nib.Nifti1Image(MKad_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKad_fit', Number_runs))
nib.Nifti1Image(MK_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MK_fit', Number_runs))
nib.Nifti1Image(MKd_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('MKd_fit', Number_runs))
nib.Nifti1Image(uFA_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('uFA_fit', Number_runs))
nib.Nifti1Image(S_I_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('S_I_fit', Number_runs))
nib.Nifti1Image(S_A_fit, affine).to_filename('Simulation_WM_{}_withSNR_{}.nii'.format('S_A_fit', Number_runs))
"""




'prepare simulation for gray matter'
noise_sigma_mask = sigma_map * gm_mask
md_mask = md_map * gm_mask
fa_mask = fa_map * gm_mask

md_gt_gray = np.mean(md_map[np.where(gm_mask ==1)])
print('gray matter mean MD from data', md_gt_gray)
md_gt_std_gray = np.std(md_map[np.where(gm_mask ==1)])

fa_gt_gray = np.mean(fa_map[np.where(gm_mask ==1)])
print('gray matter mean FA from data', fa_gt_gray)
fa_gt_std_gray = np.std(fa_map[np.where(gm_mask ==1)])

ufa_gt_gray = np.mean(ufa_map[np.where(gm_mask ==1)])
print('gray matter mean uFA from data', ufa_gt_gray)
ufa_gt_std_gray = np.std(ufa_map[np.where(gm_mask ==1)])

N = 1000
k = 50
mu = [1., 0., 0.]
threshold=0.8
D_shape='lin'

md_dis_gm = MD_gen(N, md_gt_gray, md_gt_std_gray)
fa_dist_gm = FA_gen(N, fa_gt_gray, fa_gt_std_gray)

# generate synthetic DTD for tissue
dt_orien = DT_orientation(N, k, mu, threshold)
dt_evecs = DT_evecs(N, dt_orien)
dt_evals = DT_evals(D_shape, md_dis_gm, fa_dist_gm)
dtens_gray_50 = Diffusion_Tensors_manual(dt_evecs, dt_evals)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_tensors(dtens_gray_50[0:100], fig, ax, factor=2)
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plt.title('Gray matter DTD, k={}'.format(k), fontsize=30)
plt.show()



sig_b0_unnormalized_gray_50 = data_signal[:, :, :, 0] * gm_mask
S0_gray_50 = np.mean(sig_b0_unnormalized_gray_50[np.where(sig_b0_unnormalized_gray_50 != 0)])


'run simulation for wm without snr'
Number_runs = 100

V_MD2_fit = np.zeros((Number_runs))
V_iso2_fit = np.zeros((Number_runs))
V_shear2_fit = np.zeros(( Number_runs))
MD_fit = np.zeros((Number_runs))
FA_fit = np.zeros((Number_runs))
V_MD_fit = np.zeros(( Number_runs))
V_iso_fit = np.zeros(( Number_runs))
V_MD1_fit = np.zeros(( Number_runs))
V_iso1_fit = np.zeros((Number_runs))
V_shear_fit  = np.zeros((Number_runs))
V_shear1_fit = np.zeros((Number_runs))
C_MD_fit = np.zeros((Number_runs))
C_mu_fit = np.zeros((Number_runs))
C_M_fit = np.zeros((Number_runs))
C_c_fit = np.zeros(( Number_runs))
MKi_fit = np.zeros((Number_runs))
MKa_fit = np.zeros(( Number_runs))
MKt_fit = np.zeros((Number_runs))
MKad_fit = np.zeros((Number_runs))
MK_fit = np.zeros(( Number_runs))
MKd_fit = np.zeros(( Number_runs))
uFA_fit = np.zeros((Number_runs))
S_I_fit = np.zeros(( Number_runs))
S_A_fit = np.zeros((Number_runs))


for i in range(Number_runs):
    #print(i)
    V_MD2_fit[i], V_iso2_fit[i], V_shear2_fit[i], MD_fit[i], FA_fit[i], V_MD_fit[i], \
    V_iso_fit[i], V_MD1_fit[i], V_iso1_fit[i], V_shear_fit[i], V_shear1_fit[i], C_MD_fit[i], \
    C_mu_fit[i], C_M_fit[i], C_c_fit[i], MKi_fit[i], MKa_fit[i], MKt_fit[i], MKad_fit[i], \
    MK_fit[i], MKd_fit[i], uFA_fit[i], S_I_fit[i], S_A_fit[i] = run_sim_noiseless(S0_gray_50, btensors, dtens_gray_50)


print('--------------------------------------------------------')
print('Gray matter MD noiseless simulation:', np.mean(MD_fit), 'p/m', np.std(MD_fit))
print('Gray matter FA noiseless simulation:', np.mean(FA_fit), 'p/m', np.std(FA_fit))
print('Gray matter uFA noiseless simulation:', np.mean(uFA_fit), 'p/m', np.std(uFA_fit))
print('--------------------------------------------------------')
MD_fit_noiseless_gray_50 = np.mean(MD_fit)
MD_fit_noiseless_std_gray_50 = np.std(MD_fit)

FA_fit_noiseless_gray_50 = np.mean(FA_fit)
FA_fit_noiseless_std_gray_50 = np.std(FA_fit)

uFA_fit_noiseless_gray_50 = np.mean(uFA_fit)
uFA_fit_noiseless_std_gray_50 = np.std(uFA_fit)




' run simulation for gray matter'

#SNR = 295.74
SNR = np.arange(150, 520, 10)

V_MD2_fit = np.zeros((len(SNR), Number_runs))
V_iso2_fit = np.zeros((len(SNR), Number_runs))
V_shear2_fit = np.zeros((len(SNR), Number_runs))
MD_fit = np.zeros((len(SNR), Number_runs))
FA_fit = np.zeros((len(SNR), Number_runs))
V_MD_fit = np.zeros((len(SNR), Number_runs))
V_iso_fit = np.zeros((len(SNR), Number_runs))
V_MD1_fit = np.zeros((len(SNR), Number_runs))
V_iso1_fit = np.zeros((len(SNR), Number_runs))
V_shear_fit  = np.zeros((len(SNR), Number_runs))
V_shear1_fit = np.zeros((len(SNR), Number_runs))
C_MD_fit = np.zeros((len(SNR), Number_runs))
C_mu_fit = np.zeros((len(SNR), Number_runs))
C_M_fit = np.zeros((len(SNR), Number_runs))
C_c_fit = np.zeros((len(SNR), Number_runs))
MKi_fit = np.zeros((len(SNR), Number_runs))
MKa_fit = np.zeros((len(SNR), Number_runs))
MKt_fit = np.zeros((len(SNR), Number_runs))
MKad_fit = np.zeros((len(SNR), Number_runs))
MK_fit = np.zeros((len(SNR), Number_runs))
MKd_fit = np.zeros((len(SNR), Number_runs))
uFA_fit = np.zeros((len(SNR), Number_runs))
S_I_fit = np.zeros((len(SNR), Number_runs))
S_A_fit = np.zeros((len(SNR), Number_runs))


for i in range(Number_runs):
    print(i)
    for j in range(len(SNR)):
        print('SNR', SNR[j])
        #V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        #V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        #C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        #MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim(wm_mask, data_signal_b0, md_map, fa_map, sigma_map, btensors, SNR[j])
        V_MD2_fit[j, i], V_iso2_fit[j, i], V_shear2_fit[j, i], MD_fit[j, i], FA_fit[j, i], V_MD_fit[j, i], \
        V_iso_fit[j, i], V_MD1_fit[j, i], V_iso1_fit[j, i], V_shear_fit[j, i], V_shear1_fit[j, i], C_MD_fit[j, i],\
        C_mu_fit[j, i], C_M_fit[j, i], C_c_fit[j, i], MKi_fit[j, i], MKa_fit[j, i], MKt_fit[j, i], MKad_fit[j, i],\
        MK_fit[j, i], MKd_fit[j, i], uFA_fit[j, i], S_I_fit[j, i], S_A_fit[j, i] = run_sim_v2(S0_gray_50, btensors, dtens_gray_50, SNR[j])
    print("--- %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(MD_fit, axis=1), yerr=np.std(MD_fit, axis=1),fmt='o')
plt.axhline(y=md_gt_gray, xmin=0, xmax=np.max(SNR), color='r',label='mean MD from fit = {} $\mu$m²/ms'.format(np.round(md_gt_gray, decimals=3)))
plt.axhline(y=MD_fit_noiseless_gray_50, xmin=0, xmax=np.max(SNR),color='k', label='mean MD from noiseless simulation = {} $\mu$m²/ms'.format(np.round(MD_fit_noiseless_gray_50, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('MD[$\mu$m²/ms]')
plt.title('MD(SNR) for a synthetic DTD of GM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(FA_fit, axis=1), yerr=np.std(FA_fit, axis=1), fmt='o')
plt.axhline(y=fa_gt_gray, xmin=0, xmax=np.max(SNR),color='r', label='mean FA from fit = {}'.format(np.round(fa_gt_gray, decimals=3)))
plt.axhline(y=FA_fit_noiseless_gray_50, xmin=0, xmax=np.max(SNR), color='k',label='mean FA from noiseless simulation = {}'.format(np.round(FA_fit_noiseless_gray_50, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('FA')
plt.title('FA(SNR) for a synthetic DTD of GM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()

plt.figure(figsize=(13, 8))
plt.errorbar(SNR, np.mean(uFA_fit, axis=1), yerr=np.std(uFA_fit, axis=1), fmt='o')
plt.axhline(y=fa_gt_gray, xmin=0, xmax=np.max(SNR),color='r', label='mean uFA from fit = {}'.format(np.round(fa_gt_gray, decimals=3)))
plt.axhline(y=uFA_fit_noiseless_gray_50, xmin=0, xmax=np.max(SNR),color='k', label='mean uFA from noiseless simulation = {}'.format(np.round(uFA_fit_noiseless_gray_50, decimals=3)))
plt.xlabel('SNR')
plt.ylabel('$\mu$FA')
plt.title('$\mu$FA(SNR) for a synthetic DTD of GM with N = {}, k = {}, mu = {}'.format(N,k, mu))
plt.legend(loc='upper right')
plt.rcParams.update({'font.size': 18})
plt.show()


#nib.Nifti1Image(MD_fit, affine).to_filename('Simulation_GM_MDfit_withSNR_{}.nii'.format(Number_runs))
#nib.Nifti1Image(FA_fit, affine).to_filename('Simulation_GM_FAfit_withSNR_{}.nii'.format(Number_runs))

"""
# save back nifti of result with same affine as "output_filename"
nib.Nifti1Image(V_MD2_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_MD2_fit', Number_runs))
nib.Nifti1Image(V_iso2_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_iso2_fit', Number_runs))
nib.Nifti1Image(V_shear2_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_shear2_fit', Number_runs))
nib.Nifti1Image(MD_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MD_fit', Number_runs))
nib.Nifti1Image(FA_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('FA_fit', Number_runs))
nib.Nifti1Image(V_MD_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_MD_fit', Number_runs))
nib.Nifti1Image(V_iso_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_iso_fit', Number_runs))
nib.Nifti1Image(V_MD1_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_MD1_fit', Number_runs))
nib.Nifti1Image(V_iso1_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_iso1_fit', Number_runs))
nib.Nifti1Image(V_shear_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_shear_fit', Number_runs))
nib.Nifti1Image(V_shear1_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('V_shear1_fit', Number_runs))
nib.Nifti1Image(C_MD_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('C_MD_fit', Number_runs))
nib.Nifti1Image(C_mu_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('C_mu_fit', Number_runs))
nib.Nifti1Image(C_M_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('C_M_fit', Number_runs))
nib.Nifti1Image(C_c_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('C_c_fit', Number_runs))
nib.Nifti1Image(MKi_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MKi_fit', Number_runs))
nib.Nifti1Image(MKa_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MKa_fit', Number_runs))
nib.Nifti1Image(MKt_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MKt_fit', Number_runs))
nib.Nifti1Image(MKad_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MKad_fit', Number_runs))
nib.Nifti1Image(MK_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MK_fit', Number_runs))
nib.Nifti1Image(MKd_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('MKd_fit', Number_runs))
nib.Nifti1Image(uFA_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('uFA_fit', Number_runs))
nib.Nifti1Image(S_I_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('S_I_fit', Number_runs))
nib.Nifti1Image(S_A_fit, affine).to_filename('Simulation_GM_{}_withSNR_{}.nii'.format('S_A_fit', Number_runs))
"""

