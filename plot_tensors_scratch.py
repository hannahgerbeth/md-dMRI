import numpy as np
import matplotlib.pyplot as plt
from Definitions import cov_mat, voigt_notation, dtd_cov_1d_data2fit_v1, DT_evecs, DT_evals, Diffusion_Tensors_manual
from Definitions import S_dis, plot_tensors, DT_orientation, FA_gen, MD_gen, voigt_notation, cov_mat, cov_mat_v2, noisy_signal

N = 100
k = 100
mu = [0., 0., 1.]
D_shape = 'lin'

# white matter
# FA_forDT = np.clip(FA_gen(N, FA_mean=0.25345958, FA_sigma=0.04768097), 0, 1)
# MD_forDT = MD_gen(N, MD_mean=1.96136104e-05, MD_sigma=2.71228043e-06)

FA_forDT = np.clip(FA_gen(N, FA_mean=0.99, FA_sigma=1e-9), 0, 1)
MD_forDT = MD_gen(N, MD_mean=1e-3, MD_sigma=1e-9)

DT_orien = DT_orientation(N, k, mu, threshold=0.5)
DT_evecs = DT_evecs(N, DT_orien)
DT_evals = DT_evals(D_shape, MD_forDT, FA_forDT)
DT = Diffusion_Tensors_manual(DT_evecs, DT_evals)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plot_tensors(DT, fig, ax, factor=10)
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
ax.set_xlabel('', size=18)
ax.set_ylabel('', size=18)
ax.set_zlabel('', size=18)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
plt.title('Sample of Diffusion Tensor Distribution\n k = {}'.format(k), size=28)
plt.rcParams.update({'font.size': 18})
plt.subplots_adjust(top=0.85,bottom=0.0,left=0.0,right=1.0,hspace=0.2,wspace=0.2)
plt.show()