import numpy as np
import os
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, reorient_btensors
from Definitions import cov_mat, voigt_notation, dtd_cov_1d_data2fit_v1, DT_evecs, DT_evals, Diffusion_Tensors_manual
from Definitions import S_dis, plot_tensors, DT_orientation, FA_gen, MD_gen, voigt_notation, cov_mat, cov_mat_v2
from Definitions import S_cum_ens, get_params, voigt_2_tensor
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit
from Definitions import noisy_signal

# this is a short simulation to check, whether the fit-function works
# take the M btensors as they were used in the latest measurement
# the data from the latest mesaurement was already fitted, but somthing's wrong with the microFA
# So now: for a white matter ROI, take the wm_mask and calculated FA_mean, FA_std, MD_mean, MD_std within roi
# take them as input values to generate N synthetic diffusion tensors
# simulate a signal with these N diffusion tensors and M btensors
# fit the simulated signal and check for the fitted MD, FA and uFA

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")


'Get Btensor-Data from txt-files'
# btensors are in scanning order, starting with b0 and with b0s after every shell
# btensors and data have same spatial orientation

# read a txt-file and bring the values into an array-format
filename = str('btens_oneB0.txt')
def readfile_btens(filename):
    a = np.loadtxt(filename)
    return np.array([np.reshape(a[i], (3, 3)) for i in range(len(a))])
    # output: ndarray (n, 3, 3)

btensors = readfile_btens(filename) # output: ndarray (n, )
print('btensor shape', btensors.shape)

btensors = btensors * 10**(-3)



os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg/results")
FA, afiine = load_data('FA_fit.nii')
MD, affine = load_data('MD_fit.nii')

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
wm_mask, affine = load_data('WM_mask_220422_final.nii')

FA_mean = np.mean(FA[np.where(wm_mask==1)]) #something in the range of 0.343164596891819
FA_std = np.std(FA[np.where(wm_mask==1)]) #something in the range of 0.06755988519876364

MD_mean = np.mean(MD[np.where(wm_mask==1)]) # something in the range of 0.008870602967858493
MD_std = np.std(MD[np.where(wm_mask==1)]) # something in the range of 0.001085571973069796



""" Set up a synthetic system of Diffusion Tensors"""
# first to check the Fit function:

def synthetic_tissue_simulation(N, k, mu, FA_mean, FA_std, MD_mean, MD_std, threshold=0.9):
    dt_orient = DT_orientation(N, k, mu, threshold)
    dt_evecs = DT_evecs(N, dt_orient)

    FA = FA_gen(N, FA_mean, FA_std)
    MD = MD_gen(N, MD_mean, MD_std)

    dt_evals = DT_evals('lin', MD, FA)

    DT = Diffusion_Tensors_manual(dt_evecs, dt_evals)
    #DT_mean = np.mean(DT, axis=0)

    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #plot_tensors(DT, fig, ax, factor=8)
    #plt.show()

    # Check MD and FA via eigenvalue decomposition: Definitions according to DTI
    #evals_mean, evecs_mean = np.linalg.eig(DT_mean)
    #MD_calc = (1/3) * np.sum(evals_mean)
    #FA_calc = np.sqrt( (evals_mean[0] - evals_mean[1])**2 / (evals_mean[0]**2 + 2*evals_mean[1]**2) )
    # -> pretty close

    #DT_voigt = voigt_notation(DT)
    #DT_voigt_mean = np.mean(DT_voigt, axis=0)
    #DT_covmat = cov_mat(DT_voigt)

    #DT_covmat_MP = cov_mat_v2(DT_voigt)
    #get_C, get_Vmd, get_Cmd, get_Cmu, get_CM, get_Cc, get_MD, get_uFA, get_FA, get_OP, get_MK, get_K_mu = get_params(DT)


    # simple signal simulation
    # S = mean ( exp(B*D) )
    S_sim = S_dis(btensors, DT)

    # cumulant approximation
    # S \aprox exp( - < B, mean(D) > + 0,5 * < B°2, C > )
    S_sim_cum = S_cum_ens(btensors, DT)


    ' Fit '
    K = 28 # number of variables that the fitfunktion has as uotput (from linear least squares fit)
    results = np.zeros(K)
    #print(results.shape)

    results = dtd_cov_1d_data2fit(S_sim, btensors, cond_limit=1e-20, clip_eps=1e-20)

    s0_convfit, d2_convfit, c4_convfit = convert_m(results)

    V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit,\
    V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit,\
    S_I_fit, S_A_fit = decode_m(d2_convfit, c4_convfit, reg=1e-4)

    print('Input MD:', MD_mean, '\n Fit_MD:', MD_fit)
    print('Input FA:', FA_mean, '\n Fit_FA:', FA_fit)
    print('Fit uFA:', uFA_fit, '\n mean of distributed FA:', np.mean(FA))

N = 1000
k1 = 100
k2 = 0.1
mu = [1., 0., 0.]

# when all tensors point along the (nearly) same direction, we expect that FA and uFA are (nearly) equal
synthetic_tissue_simulation(N, k1, mu, FA_mean, FA_std, MD_mean, MD_std, threshold=0.9)
#-> true
print('_-------------------------------------------------------------')
# # when all tensors point along different directions, we expect that FA_fit < uFA_fit
# but therefore uFA_fit = FA_input
synthetic_tissue_simulation(N, k2, mu, FA_mean, FA_std, MD_mean, MD_std, threshold=0.5)
#-> true






""" Compare data signal in WM roi with simulated signal in WM Roi"""
data, affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')
wm_mask, affine = load_data('WM_mask_220422_final.nii')
data_wm = np.mean(data[np.where(wm_mask==1)], axis=0)

results = dtd_cov_1d_data2fit(data_wm, btensors, cond_limit=1e-20, clip_eps=1e-20)
s0_convfit, d2_convfit, c4_convfit = convert_m(results)

V_MD2_fit, V_iso2_fit, V_shear2_fit, MD_fit, FA_fit, V_MD_fit, V_iso_fit, V_MD1_fit, V_iso1_fit, V_shear_fit,\
V_shear1_fit, C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MKi_fit, MKa_fit, MKt_fit, MKad_fit, MK_fit, MKd_fit, uFA_fit,\
S_I_fit, S_A_fit = decode_m(d2_convfit, c4_convfit, reg=1e-4)



N = 1000
k = 0.1
mu = [1., 0., 0.]

dt_orient = DT_orientation(N, k, mu, threshold=0.5)
dt_evecs = DT_evecs(N, dt_orient)
FA = FA_gen(N, FA_mean, FA_std)
MD = MD_gen(N, MD_mean, MD_std)
dt_evals = DT_evals('lin', MD, FA)
DT = Diffusion_Tensors_manual(dt_evecs, dt_evals)

S_sim_cum = S_cum_ens(btensors, DT)
sigma_mean_in_wmroi = 0.0007722047435225743
S_sim_cum_noisy = noisy_signal(S_sim_cum, sigma_mean_in_wmroi)[1]


plt.plot(data_wm, label='data')
plt.plot(S_sim_cum, label='simulation')
plt.plot(S_sim_cum_noisy, label='simulation noisy')
plt.legend()
plt.show()