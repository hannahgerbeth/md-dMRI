import numpy as np
import os
import numpy.linalg
import nibabel as nib
import matplotlib.pyplot as plt
from Definitions import cov_mat, voigt_notation, dtd_cov_1d_data2fit_v1, DT_evecs, DT_evals, Diffusion_Tensors_manual
from Definitions import S_dis, plot_tensors, DT_orientation, FA_gen, MD_gen, voigt_notation, cov_mat, cov_mat_v2, noisy_signal
from Definitions import S_cum_ens, get_params, voigt_2_tensor, fit_signal_ens, exp_signal, b_tensors, b_ten_orien
from Definitions_smallscripts import load_data, readfile_btens, mean_bvals, mean_signal, mean_Sb
#from Definitions_smallscripts import monoexp, monoexp_fit, bi_exp, biexp_fit, cumulant_exp, cumexp_fit
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit, dtd_cov_1d_data2fit, convert_m
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Tensor_math_MPaquette import tp, _S_ens
import scipy
from scipy.optimize import differential_evolution
import warnings
import time


' Load data, btensors, bvalues and mask '
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')  # shape (90,60,130,331)

btensors = readfile_btens('btens_oneB0.txt')
bvals = np.loadtxt('bvals_oneB0.txt')  # bvals in s/mm2

btensors = btensors * 10 ** (-3)  # ms/ym^2
bvals = bvals * 10 ** (-3)

mask, affine = load_data('mask_pad.nii')

mask_wm, affine = load_data('WM_mask_220422_final.nii')
mask_gm, affine = load_data('GM_mask_220422_final.nii')

data_wm = data_load * mask_wm[:, :, :, None]
data_gm = data_load * mask_gm[:, :, :, None]

signal_wm = np.mean(data_wm[np.where(mask_wm==1)], axis=0)
signal_gm = np.mean(data_gm[np.where(mask_gm==1)], axis=0)


# only fit dot-fraction on linear bvals
lin_idx = 6

b_mean_wm, S_mean_wm = mean_Sb(signal_wm, bvals)
bmean_lin_wm = np.hstack([0, b_mean_wm[lin_idx:]])
Smean_lin_wm = np.hstack([S_mean_wm[0], S_mean_wm[lin_idx:]])
bmean_pla_wm = np.hstack([b_mean_wm[:lin_idx]])
Smean_pla_wm = np.hstack([S_mean_wm[:lin_idx]])

b_mean_gm, S_mean_gm = mean_Sb(signal_gm, bvals)
bmean_lin_gm = np.hstack([0, b_mean_gm[lin_idx:]])
Smean_lin_gm = np.hstack([S_mean_gm[0], S_mean_gm[lin_idx:]])
bmean_pla_gm = np.hstack([b_mean_gm[:lin_idx]])
Smean_pla_gm = np.hstack([S_mean_gm[:lin_idx]])


fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
ax1.plot(np.sort(bmean_pla_wm), np.sort(Smean_pla_wm)[::-1], 'o--', label='pla')
ax1.plot(bmean_lin_wm, Smean_lin_wm, 'o--', label='lin')
ax1.set_xlabel('b-value [ms/$\mu m^2$]')
ax1.set_ylabel('normalized signal S/S$_{0}$')
ax1.set_ylim([0., 1.])
ax1.legend()
ax1.set_title('White matter ROI')

ax2 = fig.add_subplot(212)
ax2.plot(np.sort(bmean_pla_gm), np.sort(Smean_pla_gm)[::-1], 'o--', label='pla')
ax2.plot(bmean_lin_gm, Smean_lin_gm, 'o--', label='lin')
ax2.set_xlabel('b-value [ms/$\mu m^2$]')
ax2.set_ylabel('normalized signal S/S$_{0}$')
ax2.set_ylim([0., 1.])
ax2.legend()
ax2.set_title('Gray matter ROI')

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.show()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.semilogy(np.sort(bmean_pla_wm), np.sort(Smean_pla_wm)[::-1], 'o--', label='pla')
ax1.semilogy(bmean_lin_wm, Smean_lin_wm, 'o--', label='lin')
ax1.set_xlabel('b-value [ms/$\mu m^2$]')
ax1.set_ylabel('normalized signal S/S$_{0}$')
ax1.set_ylim([0.05, 1.])
ax1.grid(True, which="both", ls="-")
ax1.legend(loc='upper right')
ax1.set_title('White matter ROI')

ax2.semilogy(np.sort(bmean_pla_gm), np.sort(Smean_pla_gm)[::-1], 'o--', label='pla')
ax2.semilogy(bmean_lin_gm, Smean_lin_gm, 'o--', label='lin')
ax2.set_xlabel('b-value [ms/$\mu m^2$]')
ax2.set_ylabel('normalized signal S/S$_{0}$')
ax2.set_ylim([0.05, 1.])
ax2.grid(True, which="both", ls="-")
ax2.legend(loc='upper right')
ax2.set_title('Gray matter ROI')

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.show()



' look at dot-corrected signal'

' Load data, btensors, bvalues and mask '
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load_dotcorr, affine = load_data('data_b0_pla_lin_normalized_cliped_masked_dotcorrected.nii')  # shape (90,60,130,331)

data_wm_dc = data_load_dotcorr * mask_wm[:, :, :, None]
data_gm_dc = data_load_dotcorr * mask_gm[:, :, :, None]

signal_wm_dc = np.mean(data_wm_dc[np.where(mask_wm==1)], axis=0)
signal_gm_dc = np.mean(data_gm_dc[np.where(mask_gm==1)], axis=0)


b_mean_wm_dc, S_mean_wm_dc = mean_Sb(signal_wm_dc, bvals)
bmean_lin_wm_dc = np.hstack([0, b_mean_wm_dc[lin_idx:]])
Smean_lin_wm_dc = np.hstack([S_mean_wm_dc[0], S_mean_wm_dc[lin_idx:]])
bmean_pla_wm_dc = np.hstack([b_mean_wm_dc[:lin_idx]])
Smean_pla_wm_dc = np.hstack([S_mean_wm_dc[:lin_idx]])

b_mean_gm_dc, S_mean_gm_dc = mean_Sb(signal_gm_dc, bvals)
bmean_lin_gm_dc = np.hstack([0, b_mean_gm_dc[lin_idx:]])
Smean_lin_gm_dc = np.hstack([S_mean_gm_dc[0], S_mean_gm_dc[lin_idx:]])
bmean_pla_gm_dc = np.hstack([b_mean_gm_dc[:lin_idx]])
Smean_pla_gm_dc = np.hstack([S_mean_gm_dc[:lin_idx]])



fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
ax1.plot(np.sort(bmean_pla_wm_dc), np.sort(Smean_pla_wm_dc)[::-1], 'o--', label='pla')
ax1.plot(bmean_lin_wm_dc, Smean_lin_wm_dc, 'o--', label='lin')
ax1.set_xlabel('b-value [ms/$\mu m^2$]')
ax1.set_ylabel('normalized signal S/S$_{0}$')
ax1.set_ylim([0., 1.])
ax1.legend()
ax1.set_title('White matter ROI')

ax2 = fig.add_subplot(212)
ax2.plot(np.sort(bmean_pla_gm_dc), np.sort(Smean_pla_gm_dc)[::-1], 'o--', label='pla')
ax2.plot(bmean_lin_gm_dc, Smean_lin_gm_dc, 'o--', label='lin')
ax2.set_xlabel('b-value [ms/$\mu m^2$]')
ax2.set_ylabel('normalized signal S/S$_{0}$')
ax2.set_ylim([0., 1.])
ax2.legend()
ax2.set_title('Gray matter ROI')

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.show()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.semilogy(np.sort(bmean_pla_wm_dc), np.sort(Smean_pla_wm_dc)[::-1], 'o--', label='pla')
ax1.semilogy(bmean_lin_wm_dc, Smean_lin_wm_dc, 'o--', label='lin')
ax1.set_xlabel('b-value [ms/$\mu m^2$]')
ax1.set_ylabel('normalized signal S/S$_{0}$')
ax1.set_ylim([0.001, 1.])
ax1.grid(True, which="both", ls="-")
ax1.legend(loc='upper right')
ax1.set_title('White matter ROI')

ax2.semilogy(np.sort(bmean_pla_gm_dc), np.sort(Smean_pla_gm_dc)[::-1], 'o--', label='pla')
ax2.semilogy(bmean_lin_gm_dc, Smean_lin_gm_dc, 'o--', label='lin')
ax2.set_xlabel('b-value [ms/$\mu m^2$]')
ax2.set_ylabel('normalized signal S/S$_{0}$')
ax2.set_ylim([0.001, 1.])
ax2.grid(True, which="both", ls="-")
ax2.legend(loc='upper right')
ax2.set_title('Gray matter ROI')

plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.show()



' try the cumulant fit on dot-corrected data '
# we actually only care about white matter


def monoexp(x, b, c):
    return (1 - c) * np.exp(- b * x) + c

def monoexp_fit(data, bvals):

    # fit: f(x) = a * exp(-bx) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1
    # f(x) = (1-c) * exp(-bx) + c
    # -> S(b) = (1-df) * e^(-b*MD) + df
    # df = dot-fraction

    fit_bounds = ([0, 0], [np.inf, 1])

    # initialization
    # ln(S)/(-b) = ADC -> ADC depends on b and is a vector -> take mean of ADC to initialize MD
    # Suppress/hide the warning 'invalid value encountered in true_divide'
    np.seterr(invalid='ignore')

    tmp1 = np.log(data)/(- bvals)
    #MD_init = np.mean(tmp1[np.where(np.isnan(tmp1)==False)]) # if there are nan-value do not take them into account
    MD_init = np.mean(tmp1[1:])

    # take minimum of signal to initialize dot-fraction c
    if np.min(data)<=0:
        dot_init = 1e-5
    else:
        dot_init = np.min(data)

    init = ([MD_init, dot_init])

    def jac_monoexp(bs, *coef):
        c,b = coef

        d_c = 1-np.exp(-bs*b)
        d_b = (c-1)*bs*np.exp(-bs*b)

        return np.array([d_c, d_b]).T

    try:
        #tmp, pcov = scipy.optimize.curve_fit(monoexp, bvals, data, p0=init, bounds=fit_bounds, maxfev=3000)
        tmp, pcov = scipy.optimize.curve_fit(monoexp, bvals, data, p0=init,jac=jac_monoexp, bounds=fit_bounds, maxfev=5000)
    except RuntimeError:
        tmp = [0, 0]
        pcov = [0]
        print("Error - curve_fit failed: monoexp_fit")

    return tmp, pcov, init

def bi_exp(b, MD1, MD2, frac1, dotfrac):
    # E0 = (1-dotfrac)*[frac1*exp(-bs*MD1) + (1-frac1)*exp(-bs*MD2)] + dotfrac
    # 0 <= frac1, dotfrac <= 1
    # dotfrac + (1-dotfrac)*frac1 + (1-dotfrac)(1-frac1) = 1
    return (1 - dotfrac) * (frac1 * np.exp(-b * MD1) + (1 - frac1) * np.exp(-b * MD2)) + dotfrac

def biexp_fit(data, bvals, para_mono):

    # fit: f(x) = a1 * exp(-b1x) + a2 * exp(-b2x) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1
    # f(x) = (1-c) * [a exp(-b1x) + (1-a) exp(-b2x)] + c
    # -> S(b) = frac_fast * e^(-bD_fast) + frac_slow e^(-bD_slow) + dotfrac

    fit_bounds = ([0, 0, 0, 0], [np.inf, np.inf, 1, 1])

    # mono-exp fit to get initial guesses -> gives df, MD1
    # set initial guesses: MD1 = MD1_mono, MD2=MD1_mono/2, df = df_mono,
    # f(x) = (1-c) * exp(-b1x) + (1-c-a) exp(-b2x) + c
    # -> S(b) = frac_fast * e^(-bD_fast) + frac_slow e^(-bD_slow) + dotfrac
    # initial guesses: MD1 = tmp_init[0], MD2=tmp_init[0]/2, frac1=tmp_init[1], dotfrac=tmp_init[1]/2
    init = ([para_mono[0], para_mono[0]/2, para_mono[1]/2, para_mono[1]])

    def jac_biexp(bs, *coef):
        MD1, MD2, frac1, dotfrac = coef

        d_MD1 = (dotfrac - 1)*frac1*bs*np.exp(-bs*MD1)
        d_MD2 = -(dotfrac -1)*(frac1 -1)*bs*np.exp(-MD2*bs)
        d_frac1 = (1-dotfrac)*(np.exp(-MD1*bs) - np.exp(-MD2*bs))
        d_dotfrac = - frac1*np.exp(-MD1*bs) + (frac1-1)*np.exp(-MD2*bs) +1

        return np.array([d_MD1, d_MD2, d_frac1, d_dotfrac]).T

    try:
        #tmp, pcov = scipy.optimize.curve_fit(bi_exp, bvals, data, p0=init, bounds=fit_bounds, maxfev=3000)
        tmp, pcov = scipy.optimize.curve_fit(bi_exp, bvals, data, p0=init, jac=jac_biexp, bounds=fit_bounds,maxfev=10000)
    except RuntimeError:
        print("Error - curve_fit failed: biexp_fit")
        tmp = [0, 0, 0, 0]
        pcov = [0]

    return tmp, pcov, init


#md_latest_init, mk_latest_init, df_latest_init = 0, 0, 0

def cumulant_exp(x, MD, MK, c):
    #global md_latest_init
    #md_latest_init = MD
    #global mk_latest_init
    #mk_latest_init = MK
    #global df_latest_init
    #df_latest_init = c
    return (1 - c) * np.exp(- MD * x + 0.5 * MK * x ** 2) + c
    # return np.exp(- a*x + 0.5 * b**2 * x) + c

def cumexp_fit(data, bvals, para_mono):

    # fit: f(x) = (1-c) * exp(-b MD + 0.5 b^2 MK) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1

    fit_bounds = ([0, -np.inf, 0], [np.inf, np.inf, 1])

    # S = exp(- b*MD + 0.5*b**2 * MK)
    # ln(S) = - b*MD + 0.5*b**2 * MK
    # ln(S)/(-b) = b*0.5*MK - MD -> a = 0.5 MK, b = -MD
    y = np.log(data)/bvals
    x = bvals
    # remove possible (first digit) nan-values by just fitting tmp1[:1]
    a,b = np.polyfit(x[1:],y[1:],1)
    #
    MD_init = - b
    MK_init_tmp = a/0.5

    MK_init = min(MK_init_tmp, 2 * MD_init * (x[-1] - x[-2]) / (x[-1] ** 2 - x[-2] ** 2))

    # take dotfrac_init from mono-exp fit
    df_init = para_mono[1]

    init = ([MD_init, MK_init, df_init])

    def jac_cumulant(bs, *coef):
        MD, MK, c = coef

        d_MD = (c-1)*bs*np.exp(-bs*(MD-0.5*MK*bs))
        d_MK = -0.5 * (c-1) * bs**2 *np.exp(-bs*(MD-0.5*MK*bs))
        d_c = 1-np.exp(bs*(0.5*MK*bs - MD))

        return np.array([d_MD, d_MK, d_c]).T

    #plt.plot(data)
    #plt.plot(np.exp(- bvals * MD_init + 0.5 * bvals ** 2 * MK_init))
    #plt.show()


    try:
        #tmp, pcov = scipy.optimize.curve_fit(cumulant_exp, bvals, data, p0=init, bounds=fit_bounds,maxfev=5000)
        tmp, pcov = scipy.optimize.curve_fit(cumulant_exp, bvals, data, p0=init, jac=jac_cumulant, bounds=fit_bounds, maxfev=100000)
    except RuntimeError:
        print("Error - curve_fit failed: cumexp_fit")
        tmp = [0, 0, 0]
        pcov = [0]

    return tmp, pcov, init


# fit monoexp to initialize the cumulant
params_mono, pcov, init_mono = monoexp_fit(Smean_lin_wm, bmean_lin_wm)
# fit biexponential
params_biexp, pcov, init_biexp = biexp_fit(Smean_lin_wm, bmean_lin_wm, params_mono)
# fit cumulant
params_cumulant, pcov, init_cumulant = cumexp_fit(Smean_lin_wm, bmean_lin_wm, params_mono)

# fit monoexp to initialize the cumulant
params_mono_dc, pcov, init_mono = monoexp_fit(Smean_lin_wm_dc, bmean_lin_wm_dc)
params_biexp_dc, pcov, init_biexp = biexp_fit(Smean_lin_wm_dc, bmean_lin_wm_dc, params_mono_dc)
params_cumulant_dc, pcov, init_cumulant = cumexp_fit(Smean_lin_wm_dc, bmean_lin_wm_dc, params_mono_dc)



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


ax1.semilogy(bmean_lin_wm, Smean_lin_wm, label='mean linear data')
ax1.semilogy(bmean_lin_wm, monoexp(bmean_lin_wm, *params_mono), label='monoexponential approach')
ax1.semilogy(bmean_lin_wm, bi_exp(bmean_lin_wm, *params_biexp), label='biexponential approach')
ax1.semilogy(bmean_lin_wm, cumulant_exp(bmean_lin_wm, *params_cumulant), label='cumulant approach')
ax1.set_xlabel('b-value [ms/$\mu m^2$]')
ax1.set_ylabel('normalized signal S/S$_{0}$')
ax1.set_ylim(0.3, 1)
ax1.grid(which='both', axis='both')
ax1.set_title('before dot-compartment correction')
ax1.legend(loc='upper right')

ax2.semilogy(bmean_lin_wm_dc, Smean_lin_wm_dc, label='mean linear data')
ax2.semilogy(bmean_lin_wm_dc, monoexp(bmean_lin_wm_dc, *params_mono_dc), label='monoexponential approach')
ax2.semilogy(bmean_lin_wm_dc, bi_exp(bmean_lin_wm_dc, *params_biexp_dc), label='biexponential approach')
ax2.semilogy(bmean_lin_wm_dc, cumulant_exp(bmean_lin_wm_dc, *params_cumulant_dc), label='cumulant approach')
ax2.set_xlabel('b-value [ms/$\mu m^2$]')
ax2.set_ylabel('normalized signal S/S$_{0}$')
ax2.set_ylim(0.01, 1)
ax2.grid(which='both', axis='both')
ax2.set_title('after dot-compartment correction')
ax2.legend(loc='upper right')

plt.subplots_adjust(top=0.8,bottom=0.2,left=0.2,right=0.8,hspace=0.2,wspace=0.2)
plt.rcParams.update({'font.size': 14})
plt.suptitle('Fitted diffusion models on the average signal over a white matter ROI')
plt.show()

