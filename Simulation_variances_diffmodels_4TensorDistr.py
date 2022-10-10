import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from Definitions import DT_evecs, DT_evals, Diffusion_Tensors_manual, noisy_signal, _S_simple
from Definitions_smallscripts import mean_Sb, biexp_fit, monoexp_fit, cumexp_fit

# set a seed to get reproducible results
np.random.seed(0)

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

MD_input, MDfac_input, FA_input, FAfac_input = 0.6e-3, 3., 0.9, 3. # all tensors are different

DTs = DT_pop(MD_input, MDfac_input, FA_input, FAfac_input)
print('MD', MD_input, 'MDfac', MDfac_input, 'FA', FA_input, 'FAfac', FAfac_input)

# check whether MD and FA values from 4-Tensor-Distribution match the input values
w,v = np.linalg.eig(DTs)
MDs = np.sum(w, axis=1)/3
FAs = np.sqrt((1.5 * ( (w[:, 0] - MDs)**2 + 2 * (w[:, 1] - MDs)**2))/(np.sum(w**2, axis=1)))
print('MDs:',MDs)
print('FAs:',FAs)

'Generate fake b-tensors'
b_vals_gen= np.arange(0, 2000, 100)

n_bt_orien = 100
btenorien=np.full((100,3), [1., 0., 0.])
fk_bt = []
for i in range(len(b_vals_gen)):
    #fk_bt.append(b_tensors(n_bt_orien, b_vals[i], btenorien, B_shape=0))
    fk_bt.append(np.full((100,3,3), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])*b_vals_gen[i])
fake_bt = np.concatenate(np.asarray(fk_bt), axis=0)

b_vals = np.linalg.eig(fake_bt)[0][:, 0]
S = np.clip(_S_simple(fake_bt, DTs), 0,1)

dot_frac = 0.1 # 10 dot-fraction
dot = np.clip((1 - dot_frac) * S + dot_frac, 0., 1.) # signal with dot-fraction

#noise_sigma = 0.01 # Gaussian noise variance
noise_sigma = 0.1
data_sim_dot = np.clip(noisy_signal(dot, sigma=noise_sigma)[1], 0., 1.)  # signal with Gaussian noise


' fit the signal over 1000 iterations '
iterations = 1000

print('monoexponential fit')
dot_frac_mono = np.zeros((iterations))
params_mono = np.zeros((iterations, 2))
init_mono = np.zeros((iterations, 2))
for i in range(iterations):
    # generate noisy data with a certain dot-fraction
    data_sim_dot = np.clip(noisy_signal(dot, sigma=noise_sigma)[1], 0., 1.)  # gaussian noise

    # calculate mean signal and b-value to get the exponentially decaying S(b)-curve
    bmean, smean = mean_Sb(data_sim_dot, b_vals)

    # normalize the signal to 1, since corruption with noise and averaging can lead to a s/s0 not equal to 1.
    smean_norm = smean/smean[0]

    # fit method
    tmp, pcov, init_mono[i] = monoexp_fit(smean_norm, bmean)

    # save fitted parameters
    dot_frac_mono[i] = tmp[1]
    params_mono[i] = tmp


print('biexponential fit')
dot_frac_biexp = np.zeros((iterations))
params_biexp = np.zeros((iterations, 4))
init_bi = np.zeros((iterations, 4))
for i in range(iterations):
    data_sim_dot = np.clip(noisy_signal(dot, sigma=noise_sigma)[1], 0., 1.)  # gaussian noise
    bmean, smean = mean_Sb(data_sim_dot, b_vals)
    smean_norm = smean / smean[0]
    para_mono_for_init = params_mono[i]
    tmp, pcov, init_bi[i] = biexp_fit(smean_norm, bmean, para_mono_for_init)
    dot_frac_biexp[i] = tmp[3]
    params_biexp[i] = tmp


print('cumulant model fit')
dot_frac_double_dotmod= np.zeros((iterations))
params_double_dotmod = np.zeros((iterations, 3))
init_cumulant = np.zeros((iterations, 3))
for i in range(iterations):
    data_sim_dot = np.clip(noisy_signal(dot, sigma=noise_sigma)[1], 0., 1.)  # gaussian noise
    bmean, smean = mean_Sb(data_sim_dot, b_vals)
    smean_norm = smean / smean[0]
    para_mono_for_init = params_mono[i]
    tmp, pcov, init_cumulant[i] = cumexp_fit(smean_norm, bmean, para_mono_for_init)
    dot_frac_double_dotmod[i] = tmp[2]
    params_double_dotmod[i] = tmp


' Make a histogram to show model variances in estimated dot-fraction '
plt.figure(figsize=(10, 8), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.hist(dot_frac_mono, bins=100, range=(0.05,0.2), color='blue', edgecolor='k', alpha=0.65, label='monoexponential fit')
plt.hist(dot_frac_biexp, bins=100, range=(0.05,0.2), color='orange', edgecolor='k', alpha=0.65, label='biexponential fit')
plt.hist(dot_frac_double_dotmod, bins=100, range=(0.05,0.2), color='green', edgecolor='k', alpha=0.65, label='cumulant fit')
plt.axvline(dot_frac, color='k', linestyle='dashed', linewidth=1, label='true dot-fraction')
plt.xlabel('estimated dot-fraction', fontsize=22)
plt.ylabel('count', fontsize=22)
plt.legend()
plt.title('Comparison of model fits: ground truth df = {}, $\sigma$ = {} '.format(dot_frac, noise_sigma))
#plt.subplots_adjust(top=0.8,bottom=0.3,left=0.3,right=0.7,hspace=0.2,wspace=0.2)
plt.show()