import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from Definitions import DT_evecs, DT_evals, Diffusion_Tensors_manual, plot_tensors, noisy_signal, get_params, _S_simple
from Definitions_smallscripts import mean_Sb, monoexp_fit, biexp_fit, cumexp_fit, monoexp, bi_exp, cumulant_exp
import scipy
from scipy.optimize import differential_evolution


# generate a 4-Tensor-Distribution
# -> with this fairly simple distribution we can test the fitting on a variety of cases/tissue types
# T1 has FA and MD, T2 has FA/FAfac MD, T3 has FA MD*MDfac, T4 has FA/FAfac MD*MDfac
def DT_pop(MD,MDfac,FA,FAfac):
    N = 4
    k = 100
    mu = [1., 0., 0.]
    D_shape = 'lin'

    #DT_orien = DT_orientation(N, k, mu, threshold=0.9)
    #DT_orien = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    DT_orien = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    DTev = DT_evecs(N, DT_orien)
    #print(DTev)

    MD_tmp = np.array([MD, MD, MD*MDfac, MD*MDfac])
    FA_tmp = np.clip(np.array([FA, FA/FAfac, FA, FA/FAfac]), 0, 1)

    DTevals = DT_evals(D_shape, MD_tmp, FA_tmp)
    #print(DTevals)

    return Diffusion_Tensors_manual(DTev, DTevals)


' Generate tensors for various cases of tissue types '
#MD_input, MDfac_input, FA_input, FAfac_input = 0.75e-3, 2., 0.7, 1.5 #start
#MD_input, MDfac_input, FA_input, FAfac_input = 0.8e-3, 1., 0.7, 1.5 # same sizes
#MD_input, MDfac_input, FA_input, FAfac_input = 0.75e-3, 1., 0.7, 1. # all tensors are the same
#MD_input, MDfac_input, FA_input, FAfac_input = 0.5e-3, 3., 0.7, 1. # same shapes
MD_input, MDfac_input, FA_input, FAfac_input = 0.6e-3, 3., 0.9, 3. # all tensors are different

DTs = DT_pop(MD_input, MDfac_input, FA_input, FAfac_input) # generate diffusion tensors
print('MD', MD_input, 'MDfac', MDfac_input, 'FA', FA_input, 'FAfac', FAfac_input)

# check for variations in the 4-Tensor-Distribution
C, V_md, C_md, C_mu, C_M, Cc, MD, FA_mu, FA, OP, MK, K_mu = get_params(DTs)


# check whether MD and FA values from 4-Tensor-Distribution match the input values
w,v = np.linalg.eig(DTs)
MDs = np.sum(w, axis=1)/3
FAs = np.sqrt((1.5 * ( (w[:, 0] - MDs)**2 + 2 * (w[:, 1] - MDs)**2))/(np.sum(w**2, axis=1)))
print('MDs:',MDs)
print('FAs:',FAs)


' Generate fake b-tensors '
b_vals_gen= np.arange(0, 2000, 100) # s/mm^2

n_bt_orien = 100
btenorien=np.full((100,3), [1., 0., 0.])
fk_bt = []
for i in range(len(b_vals_gen)):
    #fk_bt.append(b_tensors(n_bt_orien, b_vals[i], btenorien, B_shape=0))
    fk_bt.append(np.full((100,3,3), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])*b_vals_gen[i])
fake_bt = np.concatenate(np.asarray(fk_bt), axis=0)

' calculate b-values from fake b-tensors '
b_vals = np.linalg.eig(fake_bt)[0][:, 0]

' Generate a signal '
S = np.clip(_S_simple(fake_bt, DTs), 0,1)

# exponentially decayying S(b)-curve
bvals_mean, signal_mean = mean_Sb(S, b_vals)
plt.plot(bvals_mean, signal_mean)
plt.title('Mean signal with b-value')
plt.show()

# pure signal with visible b-tensor shells
plt.plot(S)
plt.title('Pure signal with fake b-tensors')
plt.show()


' optional visualization of the S(b)-curves for different tissue types '
def quick_plot_signal(MD_in, MDfac_in, FA_in, FAfac_in):
    DTs = DT_pop(MD_in, MDfac_in, FA_in, FAfac_in)

    C, V_md, C_md, C_mu, C_M, Cc, MD, FA_mu, FA, OP, MK, K_mu = get_params(DTs)
    #print('C', C, '\n V_md', V_md, '\n C_md', C_md, '\n C_mu', C_mu, '\n C_M', C_M, '\n Cc', Cc, '\n MD', MD,
    #      '\n FA_mu', FA_mu, '\n FA', FA, '\n OP', OP, '\n MK', MK, '\n K_mu', K_mu)

    # check
    w, v = np.linalg.eig(DTs)
    MDs = np.sum(w, axis=1) / 3
    FAs = np.sqrt((1.5 * ((w[:, 0] - MDs) ** 2 + 2 * (w[:, 1] - MDs) ** 2)) / (np.sum(w ** 2, axis=1)))

    'Generate fake b-tensors'
    b_vals_gen = np.arange(0, 2000, 100)
    n_bt_orien = 100
    fk_bt = []
    for i in range(len(b_vals_gen)):
        fk_bt.append(np.full((n_bt_orien, 3, 3), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) * b_vals_gen[i])
    fake_bt = np.concatenate(np.asarray(fk_bt), axis=0)
    b_vals = np.linalg.eig(fake_bt)[0][:, 0]

    S = np.clip(_S_simple(fake_bt, DTs), 0, 1)

    bvals_mean, signal_mean = mean_Sb(S, b_vals)
    #plt.title('$V_{md}$ = {}, C_md = {},C_mu = {}, Cc = {}, FAmu = {}, MK = {}'.format(V_md,  C_md, C_mu, Cc,  FA_mu, MK))
    return plt.plot(bvals_mean, signal_mean, label='MDs:{} \n FAs:{} \n $C_md$ = {}, $C_\mu$ = {}, $\mu$FA = {}, MK = {}'.format(np.round(MDs, decimals=5), np.round(FAs,decimals=3), np.round(C_md, decimals=5), np.round(C_mu,decimals=3), np.round(FA_mu,decimals=3), np.round(MK,decimals=3)))

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 18})
quick_plot_signal(0.75e-3, 2, 0.7, 1.5) #start
quick_plot_signal(0.6e-3, 3., 0.9, 3.) # all tensors are different
quick_plot_signal(0.8e-3, 1., 0.7, 1.5) # same sizes
quick_plot_signal(0.5e-3, 3., 0.7, 1.) # same shapes
quick_plot_signal(0.75e-3, 1., 0.7, 1.) # all tensors are the same
plt.xlabel('b-value [s/mm$^2$]', fontsize=18)
plt.ylabel('S/S$_0$', fontsize=18)
plt.title('S(b) curves for various cases of the 4-Tensor-Distribution')
plt.subplots_adjust(top=0.8,bottom=0.2,left=0.2,right=0.8,hspace=0.2,wspace=0.2)
plt.legend(fontsize=14)
plt.show()



' Check ground truth '

para_mono, pcov, init_monoexp = monoexp_fit(signal_mean, bvals_mean)
para_bi, pcov, init_biexp = biexp_fit(signal_mean, bvals_mean, para_mono)
para_dotmod, pcov, init_cumulantexp = cumexp_fit(signal_mean, bvals_mean, para_mono)

# plot the true signal as well as its model fits
plt.plot(bvals_mean, signal_mean, label='S, df = {}'.format(0))
plt.plot(bvals_mean, monoexp(bvals_mean, *para_mono), label='S monoexp, df = {}'.format(para_mono[1]))
plt.plot(bvals_mean, bi_exp(bvals_mean, *para_bi), label='S bi-exp, df = {}'.format(para_bi[3]))
plt.plot(bvals_mean, cumulant_exp(bvals_mean, *para_dotmod), label='S cumulant, df = {}'.format(para_dotmod[2]))
plt.legend()
plt.show()


' corrupt the signal with noise and dot_frac '
dot_frac = np.arange(1e-4, 0.2, 0.02) # dot-fraction
noise_sigma = np.linspace(0.005, 0.1, 20) # Gaussian noise sigma

data_sim = S.copy()
# the simulated data is based on the signal that includes the dot-fraction
print('corrupt data with noise and dot-fraction')
data_sim_dot = np.zeros((len(dot_frac), len(noise_sigma), len(data_sim)))
for i in range(len(dot_frac)):
    for j in range(len(noise_sigma)):
        dot = np.clip((1 - dot_frac[i]) * data_sim + dot_frac[i], 0., 1.)
        # add noise based on estimated noise-sigmas from noise map sigma = 8e-4, use Gaussian noise (SNR high enough)
        data_sim_dot[i, j] = np.clip(noisy_signal(dot, sigma=noise_sigma[j])[1], 0., 1.)  # gaussian noise



print('monoexponential fit')
dot_frac_mono = np.zeros((len(dot_frac), len(noise_sigma)))
params_mono = np.zeros((len(dot_frac), len(noise_sigma), 2))
init_mono = np.zeros((len(dot_frac), len(noise_sigma), 2))
for i in range(len(dot_frac)):
    for j in range(len(noise_sigma)):
        bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
        smean_norm = smean/smean[0]
        tmp, pcov, init_mono[i,j] = monoexp_fit(smean_norm, bmean)
        dot_frac_mono[i, j] = tmp[1]
        params_mono[i,j] = tmp


print('biexponential fit')
dot_frac_biexp = np.zeros((len(dot_frac), len(noise_sigma)))
params_biexp = np.zeros((len(dot_frac), len(noise_sigma), 4))
init_bi = np.zeros((len(dot_frac), len(noise_sigma), 4))
for i in range(len(dot_frac)):
    for j in range(len(noise_sigma)):
        bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
        smean_norm = smean / smean[0]
        para_mono_for_init = params_mono[i,j]
        tmp, pcov, init_bi[i,j] = biexp_fit(smean_norm, bmean, para_mono_for_init)
        dot_frac_biexp[i, j] = tmp[3]
        params_biexp[i,j] = tmp


print('cumulant model fit')
dot_frac_double_dotmod= np.zeros((len(dot_frac), len(noise_sigma)))
params_double_dotmod = np.zeros((len(dot_frac), len(noise_sigma), 3))
init_cumulant = np.zeros((len(dot_frac), len(noise_sigma), 3))
for i in range(len(dot_frac)):
    for j in range(len(noise_sigma)):
        bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
        smean_norm = smean / smean[0]
        para_mono_for_init = params_mono[i, j]
        tmp, pcov, init_cumulant[i,j] = cumexp_fit(smean_norm, bmean, para_mono_for_init)
        dot_frac_double_dotmod[i, j] = tmp[2]
        params_double_dotmod[i, j] = tmp


' plot the results '
fig = plt.figure()
plt.rcParams.update({'font.size': 12})
ax = fig.add_subplot(221)
bmean, smean = mean_Sb(data_sim_dot[0, 0], b_vals)
smean_norm = smean / smean[0]
ax.plot(bmean, smean_norm, label='data, df=0' )
ax.plot(bmean, monoexp(bmean, *params_mono[0,0]), label='monoexp, df={}'.format(params_mono[0,0][1]))
ax.plot(bmean, bi_exp(bmean, *params_biexp[0,0]), label='biexp, df={}'.format(params_biexp[0,0][3]))
ax.plot(bmean, cumulant_exp(bmean, *params_double_dotmod[0,0]), label='cumulant, df={}'.format(params_double_dotmod[0,0][2]))
ax.set_xlabel('b-value [s/$mm^2$]')
ax.set_ylabel('S/S$_0$')
ax.set_title('Starting point: df=0, $\sigma$=0')
ax.legend()

ax=fig.add_subplot(223)
bmean, smean = mean_Sb(data_sim_dot[9, 19], b_vals)
smean_norm = smean / smean[0]
ax.plot(bmean, smean_norm, label='data, df=0.18' )
ax.plot(bmean, monoexp(bmean, *params_mono[9,19]), label='monoexp, df={}'.format(params_mono[9,19][1]))
ax.plot(bmean, bi_exp(bmean, *params_biexp[9,19]), label='biexp, df={}'.format(params_biexp[9,19][3]))
ax.set_ylim([0.0, 1.05])
ax.plot(bmean, cumulant_exp(bmean, *params_double_dotmod[9,19]), label='cumulant, df={}'.format(params_double_dotmod[9,19][2]))
ax.set_xlabel('b-value [s/$mm^2$]')
ax.set_ylabel('S/S$_0$')
ax.set_title('Maximal dotfraction and noise: df=0.18, $\sigma$=0.1 ')
ax.legend()

ax = fig.add_subplot(1,2,2, projection='3d')
plot_tensors(DTs, fig, ax, factor=10)
ax.set_title('MDs = {} \n FAs = {}'.format(np.round(MDs, decimals=5), np.round(FAs, decimals=3)))

plt.subplots_adjust(top=0.8,bottom=0.15,left=0.2,right=0.8,hspace=0.4,wspace=0.2)
plt.show()

# compare and visualize the inital guesses and the final fit of each model
def plot_signals_and_inits(data, bvals, params_mono, params_biexp, params_double_dotmod, init_mono, init_bi, init_cumulant):
    bmean, smean = mean_Sb(data, bvals)
    smean_norm = smean / smean[0]

    ax = fig.add_subplot(311)
    ax.plot(bmean, smean_norm, label='data normalized')
    ax.plot(bmean, monoexp(bmean, *init_mono), label='monoexp. init')
    ax.plot(bmean, monoexp(bmean, *params_mono), label='monoexp. fit')
    ax.set_title('monoexponential model')
    ax.legend()

    ax = fig.add_subplot(312)
    ax.plot(bmean, smean_norm, label='data normalized')
    ax.plot(bmean, bi_exp(bmean, *init_bi), label='bi-exp. init')
    ax.plot(bmean, bi_exp(bmean, *params_biexp), label='bi-exp. fit')
    ax.set_title('bi-exponential model')
    ax.legend()

    ax = fig.add_subplot(313)
    ax.plot(bmean, smean_norm, label='data normalized')
    ax.plot(bmean, cumulant_exp(bmean, *init_cumulant), label='cumulant init')
    ax.plot(bmean, cumulant_exp(bmean, *params_double_dotmod), label='cumulant fit')
    ax.set_title('cumulant model')
    ax.legend()

fig = plt.figure()
plot_signals_and_inits(data_sim_dot[0,0], b_vals, params_mono[0,0], params_biexp[0,0], params_double_dotmod[0,0], init_mono[0,0], init_bi[0,0], init_cumulant[0,0])
plt.suptitle('Visualization of model dependent initalization\n Starting point: df=0, $\sigma$=0')
plt.show()

fig = plt.figure()
plot_signals_and_inits(data_sim_dot[9,19], b_vals, params_mono[9,19], params_biexp[9,19], params_double_dotmod[9,19], init_mono[9,19], init_bi[9,19], init_cumulant[9,19])
plt.suptitle('Visualization of model dependent initalization\n Maximal dotfraction and noise: df={}, $\sigma$={}'.format(np.round(dot_frac[9], decimals=3), np.round(noise_sigma[19], decimals=3)))
plt.show()


' --- look at some error --- '
def err_model(estimate_df, true_df, method):
    # model-wise percentage errors
    # true df = array(len(dotfrac))
    # estimate_df = array(len(dotfrac), len(noise_sigma))

    if method==str('rmse'):
        tmp = np.zeros((estimate_df.shape))
        for i in range(estimate_df.shape[0]):
            for j in range(estimate_df.shape[1]):
                #RMSE
                MSE = np.square(true_df[i]-estimate_df[i,j]).mean()
                tmp[i,j] = np.sqrt(MSE)
        return tmp

    if method == str('abs_error'):
        tmp = np.zeros((estimate_df.shape))
        for i in range(estimate_df.shape[0]):
            for j in range(estimate_df.shape[1]):
                # abs dot frac error
                tmp[i,j] = np.abs(true_df[i] - estimate_df[i,j])
        return tmp

    if method == str('sign_error'):
        tmp = np.zeros((estimate_df.shape))
        for i in range(estimate_df.shape[0]):
            for j in range(estimate_df.shape[1]):
                # signed dot frac error
                tmp[i,j] = true_df[i] - estimate_df[i,j]
        return tmp

    if method == str('percentage_error'):
        tmp = np.zeros((estimate_df.shape))
        for i in range(estimate_df.shape[0]):
            for j in range(estimate_df.shape[1]):
                tmp[i, j] = (np.abs(true_df[i] - estimate_df[i, j])/estimate_df[i,j] )*100
        return tmp

dot_frac_mono_err = err_model(dot_frac_mono, dot_frac, method='abs_error')
dot_frac_biexp_err = err_model(dot_frac_biexp, dot_frac, method='abs_error')
dot_frac_double_dotmod_err = err_model(dot_frac_double_dotmod, dot_frac, method='abs_error')



' --- Same stuff but do exponential fits over some iterations --- '
iteration_number = np.arange(0, 10, 1)
#iteration_number = np.arange(0, 2, 1)

# data to be used
data_sim = S.copy() # use simulated data

dot_frac_its = []
noise_sigma_its = []
dot_frac_mono_its = []
dot_frac_biexp_its = []
dot_frac_double_dotmod_its = []

for it_num in iteration_number:
    print(it_num)

    ' -------------------- Simulate data with increaing dot-fraction & corrupt data with noise -------------------- '

    # normalized signal starts at 1, go from SNR 200 (sigma 0.005) to SNR 10 (sigma 0.1)
    noise_sigma = np.linspace(0.005, 0.1, 20)

    # dot fractions
    dot_frac = np.arange(0, 0.2, 0.02)

    # the simulated data is based on the signal that includes the dot-fraction
    data_sim_dot = np.zeros((len(dot_frac), len(noise_sigma),len(data_sim)))
    for i in range(len(dot_frac)):
        for j in range(len(noise_sigma)):
            dot = np.clip((1-dot_frac[i])*data_sim + dot_frac[i], 0,1)
            # add noise based on estimated noise-sigmas from noise map sigma = 8e-4, use Gaussian noise (SNR high enough)
            data_sim_dot[i, j] = np.clip(noisy_signal(dot, sigma=noise_sigma[j])[1], 0,1) #gaussian noise


    ' fit the dot fraction '
    ' mono exponential fit on linear shells '

    dot_frac_mono = np.zeros((len(dot_frac), len(noise_sigma)))
    params_mono = np.zeros((len(dot_frac), len(noise_sigma), 2))
    for i in range(len(dot_frac)):
        for j in range(len(noise_sigma)):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            tmp, pcov, init = monoexp_fit(smean, bmean)

            dot_frac_mono[i, j] = tmp[1]
            params_mono[i, j] = tmp

    dot_frac_biexp = np.zeros((len(dot_frac), len(noise_sigma)))
    params_biexp = np.zeros((len(dot_frac), len(noise_sigma), 4))
    for i in range(len(dot_frac)):
        for j in range(len(noise_sigma)):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            para_mono_for_init = params_mono[i, j]

            tmp, pcov, init = biexp_fit(smean, bmean, para_mono_for_init)
            dot_frac_biexp[i, j] = tmp[3]
            params_biexp[i, j] = tmp

    dot_frac_double_dotmod = np.zeros((len(dot_frac), len(noise_sigma)))
    params_double_dotmod = np.zeros((len(dot_frac), len(noise_sigma), 3))
    for i in range(len(dot_frac)):
        for j in range(len(noise_sigma)):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            para_mono_for_init = params_mono[i, j]

            tmp, pcov, init = cumexp_fit(smean, bmean, para_mono_for_init)
            dot_frac_double_dotmod[i, j] = tmp[2]
            params_double_dotmod[i, j] = tmp

    ' --- save the results --- '
    dot_frac_its.append(dot_frac)
    noise_sigma_its.append(noise_sigma)
    dot_frac_mono_its.append(dot_frac_mono)
    dot_frac_biexp_its.append(dot_frac_biexp)
    dot_frac_double_dotmod_its.append(dot_frac_double_dotmod)



' --- look at the error --- '

method = 'abs_error'
dot_frac_mono_err = []
dot_frac_biexp_err = []
dot_frac_double_dotmod_err = []
for it_num in iteration_number:
    dot_frac_mono_err.append(err_model(dot_frac_mono_its[it_num], dot_frac, method=method))
    dot_frac_biexp_err.append(err_model(dot_frac_biexp_its[it_num], dot_frac, method=method))
    dot_frac_double_dotmod_err.append(err_model(dot_frac_double_dotmod_its[it_num], dot_frac, method=method))


' ------------------------------- Make a plot ------------------------------- '
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
x_labels = []
for i in range(len(noise_sigma)):
    x_labels.append('{}'.format(np.round(noise_sigma[i], decimals=3)))
x_ticks = np.arange(0, len(noise_sigma), 1)

y_labels = []
for i in range(len(dot_frac)):
    y_labels.append('{}'.format(np.round(dot_frac[i], decimals=3)))
y_ticks = np.arange(0, len(dot_frac), 1)


fig = plt.figure()
plt.rcParams.update({'font.size': 12})
# first subplot
ax = fig.add_subplot(3, 3, 1)
im01 = ax.imshow(np.mean(dot_frac_mono_err, axis=0), cmap='viridis',vmin=0,vmax=0.1)
axins01 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes,borderpad=0)
ax.set_title('Mono-exponential fit: mean absolute error')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#second subplot
ax = fig.add_subplot(3, 3, 2)
im02 = ax.imshow(np.std(dot_frac_mono_err, axis=0), cmap='gray',vmin=0,vmax=0.025)
axins02 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes,borderpad=0)
ax.set_title('Mono-exponential fit: standard deviation')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#third plot
ax = fig.add_subplot(3,3,4)
im03=ax.imshow(np.mean(dot_frac_biexp_err, axis=0), cmap='viridis',vmin=0,vmax=0.1)
axins03 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
ax.set_title('Bi-exponential fit: mean absolute error')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#fourth subplot
ax = fig.add_subplot(3,3,5)
im04=ax.imshow(np.std(dot_frac_biexp_err, axis=0), cmap='gray',vmin=0,vmax=0.025)
axins04 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
ax.set_title('Bi-exponential fit: standard deviation')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#fifth subplot
ax = fig.add_subplot(3,3,7)
im05=ax.imshow(np.mean(dot_frac_double_dotmod_err, axis=0), cmap='viridis',vmin=0,vmax=0.1)
axins05 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
ax.set_title('Cumulant fit: mean absolute error')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#sixth plot
ax = fig.add_subplot(3,3,8)
im06=ax.imshow(np.std(dot_frac_double_dotmod_err, axis=0), cmap='gray',vmin=0,vmax=0.025)
axins06 = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
ax.set_title('Cumulant fit: standard deviation')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set(xlabel='noise $\sigma$', ylabel='dot fraction')

#tensors subplot
ax = fig.add_subplot(1,4,4, projection='3d')
plot_tensors(DTs, fig, ax, factor=10)
ax.set_title('MDs = {} \n FAs = {}'.format(np.round(MDs, decimals=5), np.round(FAs, decimals=3)))

plt.colorbar(im01, cax=axins01)
plt.colorbar(im02, cax=axins02)
plt.colorbar(im03, cax=axins03)
plt.colorbar(im04, cax=axins04)
plt.colorbar(im05, cax=axins05)
plt.colorbar(im06, cax=axins06)

plt.subplots_adjust(top=0.9,bottom=0.1,left=0.11,right=0.9,hspace=0.6,wspace=0.1)
plt.suptitle('Mean error ({}) and standard deviation of fit models\n {} iterations'.format(method,len(iteration_number)))
plt.show()



