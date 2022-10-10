import numpy as np
import matplotlib.pyplot as plt
from Definitions import FA_gen, MD_gen
from Definitions_smallscripts import load_data
import os



' load data'
'------------------------------------------------------------------------------------------------------------'
#load data
data_signal, affine = load_data('data_concatenate_lin_pla.nii')
data_signal_b0 = data_signal[:, :, :, 0]
gm_mask, affine = load_data('GMmask.nii')
wm_mask, affine = load_data('WMmask.nii')
sigma_map, affine = load_data('noise_sigmas_reshape_pad.nii')

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/211021_Heschl_Bruker_Magdeburg/simulation_debug")

md_map, affine = load_data('MD_fit.nii')
fa_map, affine = load_data('FA_fit.nii')
ufa_map, affine = load_data('uFA_fit.nii')

'-----------------------------------------------------------------------------'
'get ground truth values for white matter from a first DTD fit'

# ground truth value of white matter MD from MD map
md_gt_white = np.mean(md_map[np.where(wm_mask == 1)])
print('white matter mean MD from data', md_gt_white)
md_gt_std_white = np.std(md_map[np.where(wm_mask==1)])

# ground truth value of white matter FA from FA map
fa_gt_white = np.mean(fa_map[np.where(wm_mask ==1)])
print('white matter mean FA from data', fa_gt_white)
fa_gt_std_white = np.std(fa_map[np.where(wm_mask ==1)])

# ground truth value of white matter uFA from uFA map
ufa_gt_white = np.mean(ufa_map[np.where(wm_mask ==1)])
print('white matter mean uFA from data', ufa_gt_white)
ufa_gt_std_white = np.std(ufa_map[np.where(wm_mask ==1)])

' generate the MDs and FAs for the white matter DTD'
# chose same number of DTs in distribution as there are DTs in the white matter ROI
N = len(md_map[np.where(wm_mask==1)])

md_dis_white = MD_gen(N, md_gt_white, md_gt_std_white)
fa_dist_white = FA_gen(N, fa_gt_white, fa_gt_std_white)




'-----------------------------------------------------------------------------'
'get ground truth values for gray matter from a first DTD fit'
# ground truth value of gray matter MD from MD map
md_gt_gray = np.mean(md_map[np.where(gm_mask == 1)])
print('gray matter mean MD from data', md_gt_gray)
md_gt_std_gray = np.std(md_map[np.where(gm_mask==1)])

# ground truth value of gray matter FA from FA map
fa_gt_gray = np.mean(fa_map[np.where(gm_mask ==1)])
print('gray matter mean FA from data', fa_gt_gray)
fa_gt_std_gray = np.std(fa_map[np.where(gm_mask ==1)])

# ground truth value of gray matter uFA from uFA map
ufa_gt_gray = np.mean(ufa_map[np.where(gm_mask ==1)])
print('gray matter mean uFA from data', ufa_gt_gray)
ufa_gt_std_gray = np.std(ufa_map[np.where(gm_mask ==1)])

N = len(md_map[np.where(gm_mask==1)])
md_dis_gray = MD_gen(N, md_gt_gray, md_gt_std_gray)
fa_dist_gray = FA_gen(N, fa_gt_gray, fa_gt_std_gray)


""" make the histograms """

fig,axs = plt.subplots(2,2)
plt.rcParams.update({'font.size': 14})

axs[0,0].hist(md_dis_gray, label='MD in DTD', alpha = 0.5, bins=30)
axs[0,0].hist(md_map[np.where(gm_mask==1)], label='MD in ROI', alpha = 0.5, bins=30)
axs[0,0].set_title('MD and FA distributions in a \n gray matter ROI and a synthetic DTD')
axs[0,0].axvline(x=md_gt_gray, color='r', linestyle='dashed', linewidth=2, label='mean in ROI: {} $\mu$m²/ms'.format(np.round(md_gt_gray,3)))
axs[0,0].axvline(x=np.mean(md_dis_gray), color='k', linestyle='dashed', linewidth=2, label='mean in DTD: {} $\mu$m²/ms'.format(np.round(np.mean(md_dis_gray),3)))
axs[0,0].legend(loc='upper left')
axs[0,0].set_xlabel('MD [$\mu$m²/ms]', fontsize=14)
axs[0,0].set_ylabel('count', fontsize=14)

axs[0,1].hist(md_dis_white, label='MD in DTD', alpha = 0.5, bins=30)
axs[0,1].hist(md_map[np.where(wm_mask==1)], label='MD in ROI', alpha = 0.5, bins=30)
axs[0,1].set_title('MD and FA distributions in a \n white matter ROI and a synthetic DTD')
axs[0,1].axvline(x=md_gt_white, color='r', linestyle='dashed', linewidth=2, label='mean in ROI: {} $\mu$m²/ms'.format(np.round(md_gt_white,3)))
axs[0,1].axvline(x=np.mean(md_dis_white), color='k', linestyle='dashed', linewidth=2, label='mean in DTD: {} $\mu$m²/ms'.format(np.round(np.mean(md_dis_white),3)))
axs[0,1].legend(loc='upper left')
axs[0,1].set_xlabel('MD [$\mu$m²/ms]', fontsize=14)
axs[0,1].set_ylabel('count', fontsize=14)

axs[1,0].hist(fa_dist_gray, label='FA in DTD', alpha = 0.5, bins=30)
axs[1,0].hist(fa_map[np.where(gm_mask==1)], label='FA in ROI', alpha = 0.5, bins=30)
#axs[1,0].set_title('FA distributions in gray matter ROI and in synthetic DTD')
axs[1,0].axvline(x=fa_gt_gray, color='r', linestyle='dashed', linewidth=2, label='mean in ROI: {}'.format(np.round(fa_gt_gray,3)))
axs[1,0].axvline(x=np.mean(fa_dist_gray), color='k', linestyle='dashed', linewidth=2, label='mean in DTD: {}'.format(np.round(np.mean(fa_dist_gray),3)))
axs[1,0].legend(loc='upper left')
axs[1,0].set_xlabel('FA', fontsize=14)
axs[1,0].set_ylabel('count', fontsize=14)

axs[1,1].hist(fa_dist_white, label='FA in DTD', alpha = 0.5, bins=30)
axs[1,1].hist(fa_map[np.where(wm_mask==1)], label='FA in ROI', alpha = 0.5, bins=30)
#axs[1,1].set_title('FA distributions in white matter ROI and in synthetic DTD')
axs[1,1].axvline(x=fa_gt_white, color='r', linestyle='dashed', linewidth=2, label='mean in ROI: {}'.format(np.round(fa_gt_white,3)))
axs[1,1].axvline(x=np.mean(fa_dist_white), color='k', linestyle='dashed', linewidth=2, label='mean in DTD: {}'.format(np.round(np.mean(fa_dist_white),3)))
axs[1,1].legend(loc='upper left')
axs[1,1].set_xlabel('FA', fontsize=14)
axs[1,1].set_ylabel('count', fontsize=14)

plt.show()