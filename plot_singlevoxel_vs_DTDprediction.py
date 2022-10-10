import numpy as np
import os
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data, readfile_btens
from dtd_cov_MPaquette import convert_m, decode_m, dtd_cov_1d_data2fit, decode_m_v2, convert_m, dtd_cov_1d_data2fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Tensor_math_MPaquette import tp, get_metric, _S_ens, _S_ens

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")
data,affine = load_data('data_b0_pla_lin_normalized_cliped_masked.nii')

data_wm = data[30, 30, 70, :] # choose a wm voxel
data_gm = data[50, 43, 60, :] # gm voxel [50, 43, 60, :]

btensors = readfile_btens('btens_oneB0.txt')

results_wm = dtd_cov_1d_data2fit(data_wm, btensors, cond_limit=1e-20, clip_eps=1e-16)
s0_convfit_wm, d2_convfit_wm, c4_convfit_wm = convert_m(results_wm)
data_predicted_wm = _S_ens(btensors, s0_convfit_wm, d2_convfit_wm, c4_convfit_wm)

results_gm = dtd_cov_1d_data2fit(data_gm, btensors, cond_limit=1e-20, clip_eps=1e-16)
s0_convfit_gm, d2_convfit_gm, c4_convfit_gm = convert_m(results_gm)
data_predicted_gm = _S_ens(btensors, s0_convfit_gm, d2_convfit_gm, c4_convfit_gm)


fig = plt.figure()
plt.rc('font', size=16)

ax = fig.add_subplot(2,1,1)
ax.plot(data_wm, label='data', linewidth=2)
ax.plot(data_predicted_wm, label='model prediction', linewidth=2)
ax.set_title('WM voxel [30, 30, 70]')
ax.set(ylabel='normalized signal S/S$_{0}$')
ax.legend()

ax = fig.add_subplot(2,1,2)
ax.plot(data_gm, label='data', linewidth=2)
ax.plot(data_predicted_gm, label='model prediction', linewidth=2)
ax.set_title('GM voxel [50, 43, 60]')
ax.set(ylabel='normalized signal S/S$_{0}$')
ax.legend()

plt.suptitle('acquisition signal vs. model prediction for a voxel located in a ROI')
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.2,wspace=0.2)
plt.show()
