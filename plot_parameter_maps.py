import numpy as np
import scipy
import scipy.optimize
import nibabel as nib
import time
import os
import pylab as pl
import argparse
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg/results_dotcorr_biexp_nolin5_new")


# load data nifti "data_filename" with nibabel
FA = (nib.load('FA_fit.nii')).get_fdata()
FA = np.nan_to_num(FA)
muFA = (nib.load('uFA_fit.nii')).get_fdata()
MD = (nib.load('MD_fit.nii')).get_fdata()
Cmd = (nib.load('C_MD_fit.nii')).get_fdata()
Cc = (nib.load('C_c_fit.nii')).get_fdata()
Cmu = (nib.load('C_mu_fit.nii')).get_fdata()
Cm = (nib.load('C_M_fit.nii')).get_fdata()

os.chdir("C:/Users/hanna/OneDrive/Desktop/UNIHALLE/Masterarbeit/dMRI_Simulation/B-Tensor_Formalism/220422_Heschl_Bruker_Magdeburg")


temp = (nib.load('data_b0_pla_lin.nii')).get_fdata()
#temp = (nib.load('data_b0_pla_lin.nii')).get_fdata()

data = temp[:, :, :, 0] # b0 data



def subplot_conf(img_data, pos=(111), title='string', a=0, b=1):
    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    ax1 = fig.add_subplot(pos)
    im1 = ax1.imshow(img_data, cmap='gray', vmin=a, vmax=b)
    ax1.set_title(title)
    ax1.axis('off')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')#, format='%.0e')
    #fig.colorbar(im1, cax=cax, orientation='vertical', format= ticker.FuncFormatter(fmt))


fig = plt.figure(figsize=(16, 12))
plt.rcParams.update({'font.size': 18})
subplot_conf(data[:, :, 60], pos=(241), title='non-dw signal', a=0, b=0.5)
subplot_conf(MD[:, :, 60], pos=(242), title='MD $\mathrm{\cdot 10^{-3} mm^{2}/s}$', a=0, b=0.7)
subplot_conf(FA[:, :, 60], pos=(243), title='FA')
subplot_conf(Cm[:, :, 60], pos=(244), title='$C_M = FA²$', a=0, b=0.5)
subplot_conf(Cc[:, :, 60], pos=(245), title='$C_c$', a=0, b=0.5)
subplot_conf(Cmd[:, :, 60], pos=(246), title='$C_{MD}$', a=0, b=0.5)
subplot_conf(muFA[:, :, 60], pos=(247), title='$\mu FA$')
subplot_conf(Cmu[:, :, 60], pos=(248), title='$C_\mu = \mu FA²$')
plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0.2,wspace=0.2)
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 18})
subplot_conf(data[25, :, :].transpose(), pos=(141), title='non-dw signal', a=0, b=0.5)
subplot_conf(MD[25, :, :].transpose(), pos=(142), title='MD $\mathrm{\cdot 10^{-3} mm^{2}/s}$', a=0, b=0.7)
subplot_conf(FA[25, :, :].transpose(), pos=(143), title='FA')
subplot_conf(muFA[25, :, :].transpose(), pos=(144), title='$\mu$FA', a=0, b=1.5)
plt.tight_layout()
plt.show()



#fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
fig, ax = plt.subplots(2, 3)
ax[0,0].imshow(data[:, :, 40], cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[0,0].set_title('axial')
ax[0,1].imshow(data[28, :, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[0,1].set_title('sagittal')
ax[0,2].imshow(data[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[0,2].set_title('coronal')
ax[1,0].imshow(muFA[:, :, 40], cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[1,1].imshow(muFA[28, :, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[1,2].imshow(muFA[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(3, 2, figsize=(6, 12))
ax[0,0].imshow(np.pad(data[:, :, 40], ((0,0), (1,1)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[0,0].set_ylabel('axial', size=15, rotation=90)
ax[0,0].set_title('$b_{0}$')

ax[1,0].imshow(np.pad(data[28, :, :].transpose(), ((0,0), (15,15)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[1,0].set_ylabel('sagittal', size=15)

ax[2,0].imshow(data[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[2,0].set_ylabel('coronal', size=15)

ax[0,1].imshow(np.pad(muFA[:, :, 40], ((0,0), (1,1)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax[0,1].set_title('$\mu FA$')

ax[1,1].imshow(np.pad(muFA[28, :, :].transpose(), ((0,0), (15,15)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')

ax[2,1].imshow(muFA[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')

for a in ax.flat:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.spines['left'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.tick_params(left=False, bottom=False)

plt.tight_layout()
plt.show()


def ax_params(a):
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.spines['left'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.tick_params(left=False, bottom=False)

def make_colorbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


fig = plt.figure(figsize=(6, 12))

ax1 = fig.add_subplot(321)
im1 = ax1.imshow(np.pad(data[:, :, 40], ((0,0), (1,1)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax1.set_ylabel('axial', size=15, rotation=90)
ax1.set_title('$b_{0}$')
ax_params(ax1)
make_colorbar(im1, ax1)

ax2 = fig.add_subplot(323)
im2 = ax2.imshow(np.pad(data[28, :, :].transpose(), ((0,0), (15,15)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax2.set_ylabel('sagittal', size=15)
ax_params(ax2)
make_colorbar(im2, ax2)

ax3 = fig.add_subplot(325)
im3 = ax3.imshow(data[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax3.set_ylabel('coronal', size=15)
ax_params(ax3)
make_colorbar(im3, ax3)

ax4 = fig.add_subplot(322)
im4 = ax4.imshow(np.pad(muFA[:, :, 40], ((0,0), (1,1)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax4.set_title('$\mu FA$')
ax_params(ax4)
make_colorbar(im4, ax4)

ax5= fig.add_subplot(324)
im5 = ax5.imshow(np.pad(muFA[28, :, :].transpose(), ((0,0), (15,15)), 'constant', constant_values=0), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax_params(ax5)
make_colorbar(im5, ax5)

ax6 = fig.add_subplot(326)
im6 = ax6.imshow(muFA[:, 36, :].transpose(), cmap='gray', norm=plt.Normalize(vmin=0, vmax=1), origin='lower', aspect='equal')
ax_params(ax6)
make_colorbar(im6, ax6)

plt.tight_layout()
plt.show()





"""
fig = plt.figure(figsize=(16, 12))

subplot_conf(Cmu[:, 30, :], pos=(111), title='$C_\mu = \mu FA²$', a=0, b=0.000884)
plt.show()
"""