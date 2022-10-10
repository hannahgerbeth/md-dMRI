import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from Definitions import DT_evecs, DT_evals, Diffusion_Tensors_manual, noisy_signal, _S_simple, plot_tensors
from Definitions_smallscripts import mean_Sb
from Definitions_smallscripts import monoexp_fit, biexp_fit, cumexp_fit

# set a seed to get reproducible results
np.random.seed(0)

# number of 4-Tensor-Distributions to generate
N = 100

' Defnitions and functions '
# generate a 4-Tensor-Distribution
# -> with this fairly simple distribution we can test the fitting on a variety of cases/tissue types
# T1 has FA and MD, T2 has FA/FAfac MD, T3 has FA MD*MDfac, T4 has FA/FAfac MD*MDfac
def DT_pop(MD,MDfac,FA,FAfac):
    N_dt = 4
    D_shape = 'lin'

    #DT_orien = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    DT_orien = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    DTev = DT_evecs(N_dt, DT_orien)

    MD_tmp = np.array([MD, MD, MD*MDfac, MD*MDfac])
    FA_tmp = np.clip(np.array([FA, FA/FAfac, FA, FA/FAfac]), 0, 1)

    DTevals = DT_evals(D_shape, MD_tmp, FA_tmp)

    return Diffusion_Tensors_manual(DTev, DTevals)

def err_model(estimate_df, true_df, method):
    # model-wise percentage errors
    # true df = array(len(dotfrac))
    # estimate_df = array(len(dotfrac), N)

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


' Start of simulation '
MD_input = 1.0e-3
MDfac_input = np.round(np.random.uniform(low=0.5, high=2.0, size=N), decimals=6)
FA_input = np.round(np.random.uniform(low=0.5, high=0.7, size=N), decimals=6)
FAfac_input = np.round(np.random.uniform(low=1.5, high=2.5, size=N), decimals=6)

DT_dists = np.zeros((100, 4, 3, 3))
for i in range(N):
    D_shape = 'lin'
    DT_dists[i] = DT_pop(MD_input, MDfac_input[i], FA_input[i], FAfac_input[i])



'Generate fake b-tensors'
b_vals_gen= np.arange(0, 2100, 100)

n_bt_orien = 100
btenorien=np.full((100,3), [1., 0., 0.])
fk_bt = []
for i in range(len(b_vals_gen)):
    #fk_bt.append(b_tensors(n_bt_orien, b_vals[i], btenorien, B_shape=0))
    fk_bt.append(np.full((100,3,3), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])*b_vals_gen[i])
fake_bt = np.concatenate(np.asarray(fk_bt), axis=0)

b_vals = np.linalg.eig(fake_bt)[0][:, 0]


' signal from each 4-Tensor-Distribution'
S = np.zeros((N, len(b_vals)))
for i in range(N):
    S[i] = np.clip(_S_simple(fake_bt, DT_dists[i]), 0,1)

S_mean = np.zeros((N, len(b_vals_gen)))
for i in range(N):
    S_mean[i] = mean_Sb(b_vals, S[i])[0]

b_vals_mean = mean_Sb(b_vals, S[0])[1]


' --- Do exponential fits over some iterations --- '
iteration_number = np.arange(0, 10, 1) # 10 iterations

data_sim_dot_its = []

dot_frac_its = []
dot_frac_mono_its = []
dot_frac_biexp_its = []
dot_frac_cumulant_its = []

params_mono_its = []
params_biexp_its = []
params_cumulant_its = []

for it_num in iteration_number:
    print(it_num)

    ' -------------------- Simulate data with increaing dot-fraction & corrupt data with noise -------------------- '

    noise_sigma = 0.01 # Gaussian noise sigma
    dot_frac = np.arange(0.01, 0.21, 0.01)  # dot-fraction

    data_sim = S.copy()

    # the simulated data is based on the signal that includes the dot-fraction
    data_sim_dot = np.zeros((len(dot_frac), data_sim.shape[0], data_sim.shape[1]))
    for i in range(len(dot_frac)):
        for j in range(data_sim.shape[0]):
            dot = np.clip((1 - dot_frac[i]) * data_sim[j] + dot_frac[i], 0., 1.)
            data_sim_dot[i, j] = np.clip(noisy_signal(dot, sigma=noise_sigma)[1], 0., 1.)  # gaussian noise

    ' fit the dot fraction '
    ' mono exponential fit on linear shells '

    dot_frac_mono = np.zeros((len(dot_frac), data_sim.shape[0]))
    params_mono = np.zeros((len(dot_frac), data_sim.shape[0], 2))
    for i in range(len(dot_frac)):
        for j in range(data_sim.shape[0]):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            tmp, pcov, init = monoexp_fit(smean, bmean)

            dot_frac_mono[i, j] = tmp[1]
            params_mono[i, j] = tmp

    dot_frac_biexp = np.zeros((len(dot_frac), data_sim.shape[0]))
    params_biexp = np.zeros((len(dot_frac), data_sim.shape[0], 4))
    for i in range(len(dot_frac)):
        for j in range(data_sim.shape[0]):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            para_mono_for_init = params_mono[i, j]

            tmp, pcov, init = biexp_fit(smean, bmean, para_mono_for_init)
            dot_frac_biexp[i, j] = tmp[3]
            params_biexp[i, j] = tmp

    dot_frac_cumulant = np.zeros((len(dot_frac), data_sim.shape[0]))
    params_cumulant = np.zeros((len(dot_frac), data_sim.shape[0], 3))
    for i in range(len(dot_frac)):
        for j in range(data_sim.shape[0]):
            bmean, smean = mean_Sb(data_sim_dot[i, j], b_vals)
            smean = smean / smean[0]

            para_mono_for_init = params_mono[i, j]

            tmp, pcov, init = cumexp_fit(smean, bmean, para_mono_for_init)
            dot_frac_cumulant[i, j] = tmp[2]
            params_cumulant[i, j] = tmp

    ' --- save the results --- '
    data_sim_dot_its.append(data_sim_dot)

    dot_frac_its.append(dot_frac)
    dot_frac_mono_its.append(dot_frac_mono)
    dot_frac_biexp_its.append(dot_frac_biexp)
    dot_frac_cumulant_its.append(dot_frac_cumulant)

    params_mono_its.append(params_mono)
    params_biexp_its.append(params_biexp)
    params_cumulant_its.append(params_cumulant)


' --- look at the error --- '

method = 'abs_error'
dot_frac_mono_err = []
dot_frac_biexp_err = []
dot_frac_cumulant_err = []
for it_num in iteration_number:
    dot_frac_mono_err.append(err_model(dot_frac_mono_its[it_num], dot_frac, method=method))
    dot_frac_biexp_err.append(err_model(dot_frac_biexp_its[it_num], dot_frac, method=method))
    dot_frac_cumulant_err.append(err_model(dot_frac_cumulant_its[it_num], dot_frac, method=method))


' --- plot the results --- '

tmp_mono = np.concatenate(dot_frac_mono_its, axis=0)
tmp_biexp = np.concatenate(dot_frac_biexp_its, axis=0)
tmp_cum = np.concatenate(dot_frac_cumulant_its, axis=0)

fig = plt.figure()
plt.rcParams.update({'font.size': 10})
axes = []
labels = ['ground truth dot fraction','monoexp', 'biexp', 'cumulant']
for i in range(len(dot_frac)):
    ax = fig.add_subplot(5, 4, i+1)
    ax.axvline(x=dot_frac[i], color='k', ls='--', label='ground truth dot fraction = {}'.format(dot_frac[i]))
    ax.hist(tmp_mono[i], bins=30,range=(0.0,0.22), alpha=0.4, color='blue', edgecolor='k', label='mono')
    ax.hist(tmp_biexp[i], bins=30,range=(0.0,0.22), alpha=0.4, color='orange', edgecolor='k', label='biexp')
    ax.hist(tmp_cum[i], bins=30,range=(0.0,0.22), alpha=0.4, color='green', edgecolor='k', label='cumulant')
    ax.set_title('dot fraction = {}'.format(np.round(dot_frac[i], decimals=3)))
    axes.append(ax)
plt.subplots_adjust(top=0.88,bottom=0.11,left=0.05,right=0.85,hspace=0.5,wspace=0.25)
fig.legend(axes, labels=labels, loc='upper right', fontsize=10)
plt.suptitle('Absolute values of estimated dot-fractions averaged over {} iterations \n for {} 4-DT-Distributions'.format(len(iteration_number), N))
plt.show()


# look a absolute error (deviation of estimated value from ground truth value)
plt.hist(np.concatenate(np.concatenate(dot_frac_mono_err)), bins=100, range=(0.0,0.05), alpha=0.4, color='blue', edgecolor='k', label='mono')
plt.hist(np.concatenate(np.concatenate(dot_frac_biexp_err)), bins=100, range=(0.0,0.05), alpha=0.4, color='orange', edgecolor='k', label='biexp')
plt.hist(np.concatenate(np.concatenate(dot_frac_cumulant_err)), bins=100, range=(0.0,0.05), alpha=0.4, color='green', edgecolor='k', label='cumulant')
plt.title('Absolute error: deviation of estimated value from ground truth value \n Sum of all simulations: {} 4-DT-Distributions * {} iterations * {} dot-fractions'.format(N, len(iteration_number), len(dot_frac)))
plt.legend()
plt.show()



# look at absolute values
fig = plt.figure()

y_labels = []
for i in range(len(dot_frac)):
    y_labels.append('{}'.format(np.round(dot_frac[i], decimals=3)))
y_ticks = np.arange(0, len(dot_frac), 1)

plt.rcParams.update({'font.size': 10})

ax1 = fig.add_subplot(3,1,1)
im1 = ax1.imshow(np.mean(dot_frac_mono_its, axis=0), vmin=0,vmax=0.2)
ax1.set_title('monoexponential')
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels)
ax1.set_ylabel('ground truth dot fraction')
ax1.set_xlabel('4-Tensor-distributions')
plt.colorbar(im1)

ax2 = fig.add_subplot(3,1,2)
im2 = ax2.imshow(np.mean(dot_frac_biexp_its, axis=0), vmin=0,vmax=0.2)
ax2.set_title('biexponential')
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_labels)
ax2.set_ylabel('ground truth dot fraction')
ax2.set_xlabel('4-Tensor-distributions')
plt.colorbar(im2)

ax3 = fig.add_subplot(3,1,3)
im3 = ax3.imshow(np.mean(dot_frac_cumulant_its, axis=0), vmin=0,vmax=0.2)
ax3.set_title('cumulant')
ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels)
ax3.set_ylabel('ground truth dot fraction')
ax3.set_xlabel('4-Tensor-distributions')
plt.colorbar(im3)

plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.4,wspace=0.2)
plt.suptitle('Mean absolute values of estimated dot-fractions averaged over {} iterations \n for {} 4-DT-Distributions'.format(len(iteration_number), N))
plt.show()


# look at error
fig = plt.figure()

y_labels = []
for i in range(len(dot_frac)):
    y_labels.append('{}'.format(np.round(dot_frac[i], decimals=3)))
y_ticks = np.arange(0, len(dot_frac), 1)

plt.rcParams.update({'font.size': 10})

ax1 = fig.add_subplot(3,1,1)
im1 = ax1.imshow(np.mean(dot_frac_mono_err, axis=0), vmin=0,vmax=0.05)
ax1.set_title('monoexponential')
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels)
ax1.set_ylabel('ground truth dot fraction')
ax1.set_xlabel('4-Tensor-distributions')
plt.colorbar(im1)

ax2 = fig.add_subplot(3,1,2)
im2 = ax2.imshow(np.mean(dot_frac_biexp_err, axis=0), vmin=0,vmax=0.05)
ax2.set_title('biexponential')
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_labels)
ax2.set_ylabel('ground truth dot fraction')
ax2.set_xlabel('4-Tensor-distributions')
plt.colorbar(im2)

ax3 = fig.add_subplot(3,1,3)
im3 = ax3.imshow(np.mean(dot_frac_cumulant_err, axis=0), vmin=0,vmax=0.05)
ax3.set_title('cumulant')
ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels)
ax3.set_ylabel('ground truth dot fraction')
ax3.set_xlabel('4-Tensor-distributions')
plt.colorbar(im3)

plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.4,wspace=0.2)
plt.suptitle('Mean absolute error of estimated dot-fractions averaged over {} iterations \n for {} 4-DT-Distributions'.format(len(iteration_number), N))
plt.show()



' Box Plot: deviation of estimated df from ground truth df'

method = 'sign_error'
dot_frac_mono_dev = []
dot_frac_biexp_dev = []
dot_frac_cumulant_dev = []
for it_num in iteration_number:
    dot_frac_mono_dev.append(err_model(dot_frac_mono_its[it_num], dot_frac, method=method))
    dot_frac_biexp_dev.append(err_model(dot_frac_biexp_its[it_num], dot_frac, method=method))
    dot_frac_cumulant_dev.append(err_model(dot_frac_cumulant_its[it_num], dot_frac, method=method))

dot_frac_mono_dev_total = np.reshape(dot_frac_mono_dev, (-1, 1))
dot_frac_biexp_dev_total = np.reshape(dot_frac_biexp_dev, (-1, 1))
dot_frac_cumulant_dev_total = np.reshape(dot_frac_cumulant_dev, (-1, 1))

plt.hist(dot_frac_mono_dev_total, label='mono', bins=100, alpha=0.5)
plt.hist(dot_frac_biexp_dev_total, label='biexp', bins=100, alpha=0.5)
plt.hist(dot_frac_cumulant_dev_total, label='cumulant', bins=100, alpha=0.5)
plt.legend()
plt.show()

boxplot_data = np.hstack((dot_frac_mono_dev_total, dot_frac_biexp_dev_total, dot_frac_cumulant_dev_total))
boxplot_labels = ['monoexponential', 'biexponential', 'cumulant']

box = plt.boxplot(boxplot_data,  vert=True, patch_artist=True, showfliers=False, labels=boxplot_labels)
colors = ['blue', 'orange', 'green']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.grid(axis='y')
[plt.axvline(x, color = 'k', linestyle='--') for x in [1.5,2.5]]
plt.show()