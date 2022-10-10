import numpy as np
import os
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, readfile_btens, mean_Sb
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit, dtd_cov_1d_data2fit, convert_m
from Tensor_math_MPaquette import tp, _S_ens

# This file checks the fit-performance in dependence of the acquisition shell.
# Result: The DTD fit deviates the most when all shells in the linear acquisition are fitted.
#         The fit is better when only the planar and linear shells up to approximately 50.000 s/mm² are fitted.


""" Load data """
os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data_load,affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')

#data,affine = load_data('data_b0_pla_lin.nii')
wm_mask, affine = load_data('WM_mask_220422_final.nii')
wm_mask = wm_mask.astype(bool)

data = data_load[wm_mask]

""" Load btens """
btensors = readfile_btens('btens_oneB0.txt')
btensors = btensors * 10**(-3)

bvals = np.loadtxt('bvals_oneB0.txt') #bvals in s/mm2
bvals = bvals * 10**(-3) #bvals in ys/ym2

#mean_bvals, mean_signal= mean_Sb(bvals, data)

""" splitted btensors """
pla_btens = readfile_btens('pla_btens.txt')
b0 = pla_btens[0]
pla_1 = pla_btens[1:31]
pla_2 = pla_btens[31:61]
pla_3 = pla_btens[61:91]
pla_5 = pla_btens[91:121]
pla_4 = pla_btens[121:151]
#pla_btens_ordered = np.concatenate((pla_btens[:91], pla_4, pla_5), axis=0)

lin_btens = readfile_btens('lin_btens.txt')
#lin_0 = lin_btens[0]
lin_1 = lin_btens[1:31]
lin_2 = lin_btens[31:61]
lin_3 = lin_btens[61:91]
lin_4 = lin_btens[91:151]
lin_5 = lin_btens[151:181]

""" data splitted by acquisition shell """

data_0 = data[:, 0]
data_pla_1 = data[:,1:31]
data_pla_2 = data[:,31:61]
data_pla_3 = data[:,61:91]
data_pla_5 = data[:,91:121]
data_pla_4 = data[:,121:151]

data_lin_1 = data[:,151:181]
data_lin_2 = data[:,181:211]
data_lin_3 = data[:,211:241]
data_lin_4 = data[:,241:301]
data_lin_5 = data[:,301:331]



def quickfit(data, btens):
    K = 28
    results = np.zeros((data.shape[0], K))
    for idx in range(wm_mask.sum()):  # this is a loop over the number voxel in the mask
        results[idx] = dtd_cov_1d_data2fit(data[idx], btens, cond_limit=1e-20, clip_eps=1e-16)
    # in general all the loop over xyz becomes loops over only 1564 element
    # and all the intermediate data/maps/metrics etc are much much smaller (size 1564 vs 90*60*130 = 702000 so roughly 450 times less memory)

    s0_convfit = np.zeros(data.shape[0])
    d2_convfit = np.zeros((data.shape[0],6,))
    c4_convfit = np.zeros((data.shape[0], 6, 6))
    MD_fit = np.zeros(data.shape[0])
    FA_fit = np.zeros(data.shape[0])

    for idx in range(wm_mask.sum()):
        s0_convfit[idx], d2_convfit[idx], c4_convfit[idx] = convert_m(results[idx])
        # get any other parameters from the fit
        V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit[idx], FA_fit[idx], V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, \
        C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit = decode_m(d2_convfit[idx], c4_convfit[idx], reg=1e-10)

    return s0_convfit, d2_convfit, c4_convfit, MD_fit, FA_fit

def quickfit_v2(input_data, input_btensors):
    K = 28  # number of variables that the fitfunktion has as uotput (from linear least squares fit)
    results = np.zeros(input_data.shape[:3] + (K,))
    print(results.shape)

    for xyz in np.ndindex(wm_mask.shape):  # loop in N-dimension, xyz is a tuple (x,y,z)
        if wm_mask[xyz]:  # if in mask
            # results[xyz] = dtd_cov_1d_data2fit_v1(data[xyz], btensors, cond_limit=1e-10, clip_eps=1e-16) # fit
            #results[xyz] = dtd_cov_1d_data2fit(input_data[xyz], input_btensors, cond_limit=1e-20, clip_eps=1e-20)
            results[xyz] = dtd_cov_1d_data2fit(input_data[xyz], input_btensors, cond_limit=1e-20, clip_eps=1e-20)

    s0_convfit = np.zeros(input_data.shape[:3])
    d2_convfit = np.zeros(input_data.shape[:3] + (6,))
    c4_convfit = np.zeros(input_data.shape[:3] + (6, 6))
    MD_fit = np.zeros(input_data.shape[:3])
    FA_fit = np.zeros(input_data.shape[:3])

    for xyz in np.ndindex(results.shape[:3]):
        # get the ordered solution of the fit
        #s0_convfit[xyz], d2_convfit[xyz], c4_convfit[xyz] = convert_m(results[xyz])
        s0_convfit[xyz], d2_convfit[xyz], c4_convfit[xyz] = convert_m(results[xyz])
        # get any other parameters from the fit
        V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit[xyz], FA_fit[
            xyz], V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit, V_shear1_fit, C_MD_fit, \
        C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit, S_I_fit, S_A_fit = decode_m(
            d2_convfit[xyz], c4_convfit[xyz], reg=1e-10)

    return s0_convfit, d2_convfit, c4_convfit, MD_fit, FA_fit


""" build data and btensor arrays and do fit """

def sig_theo(data, mask, btens, s0, d2, c4):
    signal = np.zeros(data.shape)
    for idx in range(mask.sum()):
        signal[idx] = _S_ens(btens, s0[idx], d2[idx], c4[idx])
    return np.clip(signal, 0,1)

def RMSE_data(data, signal):
    return np.sqrt(np.square(np.subtract(data, signal))).mean()


data_11 =  np.concatenate((data_0[:, None] , data_pla_1, data_lin_1), axis=1)
btens_11 = np.concatenate((b0[None, :, :], pla_1, lin_1,), axis=0)
s0_11, d2_11, c4_11, MD_11, FA_11 = quickfit(data_11, btens_11)

signal_11 = sig_theo(data_11, wm_mask, btens_11, s0_11, d2_11, c4_11)
RMSE_11 = RMSE_data(data_11, signal_11)

#-----------------------------------------------------------------------------
data_22 = np.concatenate((data_0[:, None], data_pla_1, data_pla_2, data_lin_1, data_lin_2), axis=1)
btens_22 = np.concatenate((b0[None, :, :], pla_1, pla_2, lin_1, lin_2), axis=0)
s0_22, d2_22, c4_22, MD_22, FA_22 = quickfit(data_22, btens_22)

signal_22 = sig_theo(data_22, wm_mask, btens_22, s0_22, d2_22, c4_22)
RMSE_22 = RMSE_data(data_22, signal_22)

#-----------------------------------------------------------------------------
data_33 = np.concatenate((data_0[:, None], data_pla_1, data_pla_2, data_pla_3, data_lin_1, data_lin_2, data_lin_3), axis=1)
btens_33 = np.concatenate((b0[None, :, :], pla_1, pla_2, pla_3, lin_1, lin_2, lin_3), axis=0)
s0_33, d2_33, c4_33, MD_33, FA_33 = quickfit(data_33, btens_33)

signal_33 = sig_theo(data_33, wm_mask, btens_33, s0_33, d2_33, c4_33)
RMSE_33 = RMSE_data(data_33, signal_33)

#-----------------------------------------------------------------------------
data_44 = np.concatenate((data_0[:, None], data_pla_1, data_pla_2, data_pla_3, data_pla_4, data_lin_1, data_lin_2, data_lin_3, data_lin_4), axis=1)
btens_44 = np.concatenate((b0[None, :, :], pla_1, pla_2, pla_3, pla_4, lin_1, lin_2, lin_3, lin_4), axis=0)
s0_44, d2_44, c4_44, MD_44, FA_44 = quickfit(data_44, btens_44)

signal_44 = sig_theo(data_44, wm_mask, btens_44, s0_44, d2_44, c4_44)
RMSE_44 = RMSE_data(data_44, signal_44)

#-----------------------------------------------------------------------------
data_54 = np.concatenate((data_0[:, None], data_pla_1, data_pla_2, data_pla_3, data_pla_4, data_pla_5, data_lin_1, data_lin_2, data_lin_3, data_lin_4), axis=1)
btens_54 = np.concatenate((b0[None, :, :], pla_1, pla_2, pla_3, pla_4, pla_5, lin_1, lin_2, lin_3, lin_4), axis=0)
s0_54, d2_54, c4_54, MD_54, FA_54 = quickfit(data_54, btens_54)

signal_54= sig_theo(data_54, wm_mask, btens_54, s0_54, d2_54, c4_54)
RMSE_54 = RMSE_data(data_54, signal_54)

#-----------------------------------------------------------------------------
data_55 = np.concatenate((data_0[:, None], data_pla_1, data_pla_2, data_pla_3, data_pla_4, data_pla_5, data_lin_1, data_lin_2, data_lin_3, data_lin_4, data_lin_5), axis=1)
btens_55 = np.concatenate((b0[None, :, :], pla_1, pla_2, pla_3, pla_4, pla_5, lin_1, lin_2, lin_3, lin_4, lin_5), axis=0)
s0_55, d2_55, c4_55, MD_55, FA_55 = quickfit(data_55, btens_55)

signal_55= sig_theo(data_55, wm_mask, btens_55, s0_55, d2_55, c4_55)
RMSE_55 = RMSE_data(data_55, signal_55)

#-----------------------------------------------------------------------------

MD_mean = np.array((np.mean(MD_11), np.mean(MD_22), np.mean(MD_33), np.mean(MD_44), np.mean(MD_54), np.mean(MD_55)))
MD_std = np.array((np.std(MD_11), np.std(MD_22), np.std(MD_33), np.std(MD_44), np.std(MD_54), np.std(MD_55)))

FA_mean = np.array((np.mean(FA_11), np.mean(FA_22), np.mean(FA_33), np.mean(FA_44), np.mean(FA_54), np.mean(FA_55) ))
FA_std = np.array((np.std(FA_11), np.std(FA_22), np.std(FA_33), np.std(FA_44), np.std(FA_54), np.std(FA_55)))

bvals_list = np.round(bvals, decimals=6).tolist()
temp_bvals_list = []
for i in bvals_list:
    if i not in temp_bvals_list:
        temp_bvals_list.append(i)

bvals_mean = np.array(temp_bvals_list)
bvals_mean_pla = np.hstack([bvals_mean[:3], bvals_mean[5], bvals_mean[4]])
bvals_mean_lin = np.hstack([0, bvals_mean[6:]])

RMSE_total = np.array((RMSE_11, RMSE_22, RMSE_33, RMSE_44, RMSE_54, RMSE_55))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
x = np.array(('11', '22', '33', '44', '54', '55'))

ax1.errorbar(x[1:], MD_mean[1:], yerr=MD_std[1:], capsize=4)
ax1.set(xlabel='combination of bvalues or shells: pla+lin', ylabel='fitted MD within WM ROI')
ax1.set_title('Mean diffusivity')

ax2.errorbar(x[1:], FA_mean[1:], yerr=FA_std[1:], capsize=4)
ax2.set(xlabel='combination of bvalues or shells: pla+lin',ylabel='fitted FA within WM ROI' )
ax2.set_title('Fractional anisotropy')

ax3.plot(x[1:], RMSE_total[1:])
ax3.set(xlabel='combination of bvalues or shells: pla+lin', ylabel='root mean square error: data - predicted (fitted) signal' )
ax3.set_title('RMSE')

plt.show()

print('mean MD:', MD_mean)
print('MD std:', MD_std)
print('mean FA:', FA_mean)
print('FA std:', FA_std)

# Insert these values in the Simualtion_estimate_b_convergence_radius.py to generate a synthetic tissue based on "real" parameters
# these are White Matter parameters
# In[30]: MD_mean[4], MD_std[4]
# Out[30]: (1.961361040430951e-05, 2.712280426625316e-06)
#In [31]: FA_mean[4], FA_std[4]
#Out[31]: (0.36745855979458664, 0.07264520261837205)

mean_11 = np.mean(data_11, axis=0)
mean_11_fit = np.mean(signal_11, axis=0)
mean_11_error = np.mean(np.abs(signal_11 - data_11), axis=0)

mean_22 = np.mean(data_22, axis=0)
mean_22_fit = np.mean(signal_22, axis=0)
mean_22_error = np.mean(np.abs(signal_22 - data_22), axis=0)

mean_33 = np.mean(data_33, axis=0)
mean_33_fit = np.mean(signal_33, axis=0)
mean_33_error = np.mean(np.abs(signal_33 - data_33), axis=0)

mean_44 = np.mean(data_44, axis=0)
mean_44_fit = np.mean(signal_44, axis=0)
mean_44_error = np.mean(np.abs(signal_44 - data_44), axis=0)

mean_54 = np.mean(data_54, axis=0)
mean_54_fit = np.mean(signal_54, axis=0)
mean_54_error = np.mean(np.abs(signal_54 - data_54), axis=0)

mean_55 = np.mean(data_55, axis=0)
mean_55_fit = np.mean(signal_55, axis=0)
mean_55_error = np.mean(np.abs(signal_55 - data_55), axis=0)

fig, axs = plt.subplots(2, 3)
#x = np.array(('11', '22', '33', '44', '54', '55'))

axs[0,0].plot(mean_11, label='mean data: 11')
axs[0,0].plot(mean_11_fit, label='signal prediction')
axs[0,0].legend()
axs[0,0].set_title('11')

axs[0,1].plot(mean_22, label='mean data: 22')
axs[0,1].plot(mean_22_fit, label='signal prediction')
axs[0,1].legend()
axs[0,1].set_title('22')

axs[0,2].plot(mean_33, label='mean data: 33')
axs[0,2].plot(mean_33_fit, label='signal prediction')
axs[0,2].legend()
axs[0,2].set_title('33')

axs[1,0].plot( mean_44, label='mean data: 44')
axs[1,0].plot(mean_44_fit, label='signal prediction')
axs[1,0].legend()
axs[1,0].set_title('44')

axs[1,1].plot(mean_54, label='mean data: 54')
axs[1,1].plot(mean_54_fit, label='signal prediction')
axs[1,1].legend()
axs[1,1].set_title('54')

axs[1,2].plot(mean_55, label='mean data: 55')
axs[1,2].plot(mean_55_fit, label='signal prediction')
axs[1,2].legend()
axs[1,2].set_title('55')

fig.suptitle('Mean data and predicted signal for a WM ROI')
plt.show()

fig, axs = plt.subplots(2, 3)
axs[0,0].plot(mean_11_error, label='error: 11')
axs[0,0].legend()
axs[0,0].set_title('11')

axs[0,1].plot(mean_22_error, label='error: 22')
axs[0,1].legend()
axs[0,1].set_title('22')

axs[0,2].plot(mean_33_error, label='error: 33')
axs[0,2].legend()
axs[0,2].set_title('33')

axs[1,0].plot(mean_44_error, label='error: 44')
axs[1,0].legend()
axs[1,0].set_title('44')

axs[1,1].plot(mean_54_error, label='error: 54')
axs[1,1].legend()
axs[1,1].set_title('54')

axs[1,2].plot(mean_55_error, label='error: 55')
axs[1,2].legend()
axs[1,2].set_title('55')

plt.show()


"""--------------- How well does the 22-fit predict the 33 signal, etc.? ----------------"""
signal_22_predict_33= sig_theo(data_33, wm_mask, btens_33, s0_22, d2_22, c4_22)
signal_33_predict_44= sig_theo(data_44, wm_mask, btens_44, s0_33, d2_33, c4_33)
signal_44_predict_54= sig_theo(data_54, wm_mask, btens_54, s0_44, d2_44, c4_44)
signal_54_predict_55= sig_theo(data_55, wm_mask, btens_55, s0_54, d2_54, c4_54)

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(mean_33, label='data 33')
axs[0,0].plot(mean_33_fit, label='fit 33')
axs[0,0].plot(np.clip(np.mean(signal_22_predict_33, axis=0), 0, 1), label='fit 22 predict 33')
axs[0,0].legend()

axs[0,1].plot(mean_44, label='data 44')
axs[0,1].plot(mean_44_fit, label='fit 44')
axs[0,1].plot(np.clip(np.mean(signal_33_predict_44, axis=0), 0, 1), label='fit 33 predict 44')
axs[0,1].legend()

axs[1,0].plot(mean_54, label='data 54')
axs[1,0].plot(mean_54_fit, label='fit 54')
axs[1,0].plot(np.clip(np.mean(signal_44_predict_54, axis=0),0,1), label='fit 44 predict 54')
axs[1,0].legend()

axs[1,1].plot(mean_55, label='data 55')
axs[1,1].plot(mean_55_fit, label='fit 55')
axs[1,1].plot(np.clip(np.mean(signal_54_predict_55, axis=0),0,1), label='fit 54 predict 55')
axs[1,1].legend()

plt.suptitle('How well does a previous fit predict more shells?')

plt.show()


signal_22_predict_55 = sig_theo(data_55, wm_mask, btens_55, s0_22, d2_22, c4_22)

plt.plot(mean_55, label='data 55')
plt.plot(mean_55_fit, label='fit 55')
plt.plot(np.clip(np.mean(signal_22_predict_55[np.where(wm_mask==1)], axis=0),0,1), label='fit 22 predict 55')
plt.legend()
plt.show()
