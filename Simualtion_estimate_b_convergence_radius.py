import numpy as np
import matplotlib.pyplot as plt
from Definitions import DT_evecs, DT_evals, Diffusion_Tensors_manual
from Definitions import plot_tensors, DT_orientation, FA_gen, MD_gen
from Definitions import S_cum_ens, get_params, b_tensors, build_btens
from Definitions import b_ten_orien, noisy_signal, _S_simple
from dtd_cov_MPaquette import convert_m, decode_m, decode_m_v2, dtd_cov_1d_data2fit
from Tensor_math_MPaquette import _S_ens


""" synthetic tissue setup """
N = 1000
k = 100
mu = [1., 0., 0.]
D_shape = 'lin'

# MD and FA for white matter based on bvalue-/shell-dependend fit on data -> DTDfit_on_splitted_shells.py
#FA_forDT = np.clip(FA_gen(N, FA_mean=0.36745856, FA_sigma=0.0726452), 0, 1)
#MD_forDT = MD_gen(N, MD_mean=1.96136104e-05, MD_sigma=2.71228043e-06)

# MD and FA for dot-corrected white matter based on bvalue-/shell-dependend fit on data -> DTDfit_on_splitted_shells.py
FA_forDT = np.clip(FA_gen(N, FA_mean=0.25345958, FA_sigma=0.04768097), 0, 1)
MD_forDT = MD_gen(N, MD_mean=1.28862800e-04, MD_sigma=1.39742135e-05)

# MD and FA for dot-corrected gray matter based on bvalue-/shell-dependend fit on data -> DTDfit_on_splitted_shells.py
#FA_forDT = np.clip(FA_gen(N, FA_mean=0.09098946, FA_sigma=0.03673845), 0, 1)
#MD_forDT = MD_gen(N, MD_mean=3.38931807e-04, MD_sigma=2.94524652e-05)

DT_orien = DT_orientation(N, k, mu, threshold=0.7)
DT_evecs = DT_evecs(N, DT_orien)
DT_evals = DT_evals(D_shape, MD_forDT, FA_forDT)
DT = Diffusion_Tensors_manual(DT_evecs, DT_evals)

DT_mean = np.mean(DT, axis=0)

x = [np.random.randint(DT.shape[0]) for p in range(100)]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_tensors(DT[x], fig, ax, factor=50)
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plt.show()


""" ground truth variances from tensor distribution """

# Hannah's version
C4_dist, V_md_dist, Cmd_dist, Cmu_dist, CM_dist, Cc_dist, MD_dist, FA_mu_dist, FA_dist, OP_dist, MK_dist, K_mu_dist = get_params(DT)
print('Hannahs Parameters from DT Distribution')
print('\n Cmd', Cmd_dist,'\n Cmu',  Cmu_dist, '\n CM', CM_dist, '\n Cc', Cc_dist, '\n MD mean',  MD_dist, '\n FA mean',  FA_dist,'\n uFA_tmp', FA_mu_dist )
#print( '\n C4', C4_dist, '\n Cmd', Cmd_dist,'\n Cmu',  Cmu_dist, '\n CM', CM_dist, '\n Cc', Cc_dist, '\n MD mean',  MD_dist, '\n FA mean',  FA_dist )



""" Run as simulation """
bmax = np.array(([10, 50, 100, 300, 500, 800, 1000, 3000, 5000, 8000, 10000, 30000, 50000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 250000]))
N_it = 100 # number of iterations


C_MD_sim = np.zeros((N_it, len(bmax)))
C_mu_sim = np.zeros((N_it, len(bmax)))
C_M_sim = np.zeros((N_it, len(bmax)))
C_c_sim = np.zeros((N_it, len(bmax)))
MD_sim = np.zeros((N_it, len(bmax)))
FA_sim = np.zeros((N_it, len(bmax)))
uFA_sim = np.zeros((N_it, len(bmax)))
RMSE = np.zeros((N_it, len(bmax)))
S0_sim = np.zeros((N_it, len(bmax)))
for j in range(len(bmax)):
    print(j)
    # btensors
    b_vals = np.linspace(0, bmax[j], 5)  #
    N_bt = 30
    #bt_orien = b_ten_orien(N_bt)
    bt_lin = build_btens(N_bt, b_vals, b_ten_orien(N_bt), 0.99)
    bt_pla = build_btens(N_bt, b_vals, b_ten_orien(N_bt), -0.49)

    bt = np.concatenate((bt_lin, bt_pla), axis=0)

    # signal

    for i in range(N_it):
        #print(i)
        S = np.clip(S_cum_ens(bt, DT), 0, 1)
        #S = np.clip(_S_simple(bt, DT), 0, 1)
        S = np.clip(noisy_signal(S, sigma=0.005)[1], 0,1)  # maybe clip it to 0 and 1

        # fit
        #results = dtd_cov_1d_data2fit(S, bt, cond_limit=1e-20, clip_eps=1e-16, method='linalg')
        #s0, d2, c4 = convert_m(results)

        results = dtd_cov_1d_data2fit(S, bt, cond_limit=1e-20, clip_eps=1e-16)
        s0, d2, c4 = convert_m(results)

        #MSE = np.square(np.subtract(S, fit_signal_ens(bt, d2, c4))).mean()
        MSE = np.square(np.subtract(S, _S_ens(bt,s0, d2, c4))).mean()
        RMSE[i, j] = np.sqrt(MSE)

        C_MD_fit, C_mu_fit, C_M_fit, C_c_fit, MD_fit, FA_fit, uFA_fit = decode_m_v2(d2, c4, reg=1e-15)
        C_MD_sim[i, j] = C_MD_fit
        C_mu_sim[i, j] = C_mu_fit
        C_M_sim[i, j] = C_M_fit
        C_c_sim[i, j] = C_c_fit
        MD_sim[i, j] = MD_fit
        FA_sim[i, j] = FA_fit
        uFA_sim[i, j] = uFA_fit
        S0_sim[i, j] = s0



plt.errorbar(bmax, np.mean(S0_sim, axis=0), yerr=np.std(S0_sim, axis=0))
plt.title('S0 estimation with bval')
plt.show()


import matplotlib.ticker as mtick
fig = plt.figure()

# first subplot
ax = fig.add_subplot(2,4,1)
ax.errorbar(bmax*1e-3, np.mean(C_MD_sim, axis=0), yerr=np.std(C_MD_sim, axis=0), linewidth=2)
ax.set_title('$C_{MD}$')
ax.axhline(y=Cmd_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.set_ylim([0., 1.])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3),max(bmax*1e-3)/5))

#second subplot
ax = fig.add_subplot(2,4,2)
ax.errorbar(bmax*1e-3, np.mean(C_mu_sim, axis=0), yerr=np.std(C_mu_sim, axis=0), linewidth=2)
ax.set_title('$C_{\mu}$')
ax.axhline(y=Cmu_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.set_ylim([0., 1.])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3), max(bmax*1e-3)/5))

# third subplot
ax = fig.add_subplot(2,4,3)
ax.errorbar(bmax*1e-3, np.mean(C_M_sim, axis=0), yerr=np.std(C_M_sim, axis=0), linewidth=2)
ax.set_title('$C_{M}$')
ax.axhline(y=CM_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.set_ylim([0., 1.])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3), max(bmax*1e-3)/5))

# fourth subplot
x = [np.random.randint(DT.shape[0]) for p in range(100)]
ax = fig.add_subplot(1,4,4, projection='3d')
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plot_tensors(DT[x], fig, ax, factor=50)
ax.set_title('MD = {} mm²/s \n FA = {}'.format(np.round(MD_dist, decimals=5), np.round(FA_dist, decimals=3)))

# fifth subplot
ax = fig.add_subplot(2,4,5)
ax.errorbar(bmax*1e-3, np.mean(C_c_sim, axis=0), yerr=np.std(C_c_sim, axis=0), linewidth=2)
ax.set_title('$C_{c}$')
ax.axhline(y=Cc_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.set_ylim([0., 1.2])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3), max(bmax*1e-3)/5))

#sixth subplot
ax = fig.add_subplot(2,4,6)
ax.errorbar(bmax*1e-3, np.mean(MD_sim, axis=0), yerr=np.std(MD_sim, axis=0), linewidth=2)
ax.set_title('MD $\cdot 10^{-3}$ [mm²/s]')
ax.axhline(y=MD_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
ax.set_ylim([0., 1.e-3])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3), max(bmax*1e-3)/5))

#seventh subplot
ax = fig.add_subplot(2,4,7)
ax.errorbar(bmax*1e-3, np.mean(FA_sim, axis=0), yerr=np.std(FA_sim, axis=0),linewidth=2)
ax.set_title('FA')
ax.axhline(y=FA_dist, xmin=0, xmax=np.max(bmax), c="black", linewidth=3, zorder=0)
ax.set(xlabel='b [ms/$\mu$m²]', ylabel='parameter')
ax.set_ylim([0., 1.])
ax.xaxis.set_ticks(np.arange(0, max(bmax*1e-3), max(bmax*1e-3)/5))

plt.suptitle('Convergence radius estimation for gray matter')
#plt.suptitle('Convergence radius estimation for white matter')
plt.rc('font', size=18)
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.4,wspace=0.5)
plt.show()







