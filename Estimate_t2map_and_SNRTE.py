import numpy as np
import nibabel as nib
import time
import matplotlib.pyplot as plt
from Definitions_smallscripts import load_data
import seaborn as sns

' T2-map calculation for a given data-set + mask '

def T2_map_calc(mask, data, TE_time, range=(0, 200)):
    """
    Calculate the T2 map based of data acquisitions at different echo times (TE).
    :param mask: 3D mask of the brain
    :param data: 4D MRI data [read, phase, slice, time] (3D dataset + number of acquisitions)
    :param TE_time: list of echo times (TE) in ms
    :param range: range of T2 values to be considered in ms
    :return: T2 map (3D), estimated S0 (3D)
    """
    # input arrays: data and mask
    # mask has given shape [x, y, z]
    # data has shape (mask.shape[:3] + (K,)) with "K" values in the 4th dimension for our "K" measurements
    # 'range' optional argument returns the estimated T2-values only if its between 0 ms and 200 ms, else it is set to 0, to avoid extremely high or low (negative) values
    # if range=None all T2 values are returned, no matter what value

    if range:
        T2_fixed_range = np.zeros(data.shape[:3])
        S0_fit = np.zeros(data.shape[:3])
        for xyz in np.ndindex(data.shape[:3]):  # loop in N-dimension, xyz is a tuple (x,y,z)
            if mask[xyz]:
                a, b = np.polyfit(TE_time, np.log(data[xyz]), 1)
                temp = - 1. / a
                S0_fit[xyz] = np.exp(b)
                if temp < range[0] or temp > range[1]:
                    T2_fixed_range[xyz] = 0
                else:
                    T2_fixed_range[xyz] = temp
        return T2_fixed_range, S0_fit

    else:
        T2_map = np.zeros(data.shape[:3])
        S0_fit = np.zeros(data.shape[:3])
        for xyz in np.ndindex(data.shape[:3]):  # loop in N-dimension, xyz is a tuple (x,y,z)
            if mask[xyz]:
                a, b = np.polyfit(TE_time, np.log(data[xyz]), 1)
                T2_map[xyz] = - 1. / a
                S0_fit[xyz] = np.exp(b)
        return T2_map, S0_fit

# calculate S(TE)
def S_TE(TElist, T2map, S0_fit, mask):
    """
    Calculate the signal as a function of the echo times (TE) based on the T2 map and S0.
    :param TElist: list of echo times (TE) in ms
    :param T2map: T2 map (3D)
    :param S0_fit: estimated S0 (3D)
    :param mask: 3D mask of the brain
    :return: signal as a function of the echo times (signal.shape() + number of TE)
    """
    # TElist is a list with the TEs for different waveforms and waveform durations
    # T2 map, S0 (fit) and mask have data dimension [120, 80, 50]
    # return: S_TE = [read, phase, slice, time], where time = number of TEs
    # -> signal in dependence of different TEs, data for each TE is in the fourth dimension
    S_TE = np.zeros(T2map.shape[:3] + (len(TElist),))
    # print(S_TE.shape)

    for i in range(len(TElist)):
        for xyz in np.ndindex(T2map.shape[:3]):
            if mask[xyz]:
                S_TE[xyz][i] = S0_fit[xyz] * np.exp(- TElist[i] / T2map[xyz])
    return S_TE


def Sig2Noise_TE(Signal_TE, sigmamap, mask):
    """
    Calculate the signal-to-noise ratio (SNR) as a function of the echo times (TE) based on the signal and noise.
    :param Signal_TE: signal as a function of the echo times (signal.shape() + number of TE)
    :param sigmamap: noise map (3D)
    :param mask: 3D mask of the brain
    :return: signal-to-noise ratio (SNR) as a function of the echo times (SNR.shape() + number of TE)
    """
    SNR_TE = np.zeros(Signal_TE.shape)

    for i in range(Signal_TE.shape[3]):
        for xyz in np.ndindex(Signal_TE.shape[:3]):
            if mask[xyz]:
                SNR_TE[xyz][i] = Signal_TE[xyz][i] / sigmamap[xyz]
    return SNR_TE


' load the dMRI data '

data_img = nib.load('data_concatenate_lin_pla_normalized_new.nii')
data = data_img.get_fdata()
affine = data_img.affine


' load the white matter and gray matter masks '

mask_gm, affine = load_data('GMmask.nii')
mask_wm, affine = load_data('WMmask.nii')
mask, affine = load_data('mask_pad.nii')

sigma_map, affine = load_data('sigma_map_reshape_pad.nii')
noise_map, affine = load_data('noisemap.nii')


' load the MRI data at different TE '

data_TE40, affine = load_data('TE40ms.nii')
data_TE50, affine = load_data('TE50ms.nii')
data_TE60, affine = load_data('TE60ms.nii')
data_TE70, affine = load_data('TE70ms.nii')
data_TE100, affiner = load_data('TE100ms.nii')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
tmp = np.transpose(noise_map,(0,2,1,3))
im1 = ax1.imshow( np.rot90(tmp[:, 45, :, 0]), cmap='gray')
ax1.axis('off')
ax1.set_title('noise map',fontsize=15)
plt.colorbar(im1, ax=ax1)
im2 = ax2.imshow(np.rot90(sigma_map[:, 45, :]), cmap='gray')
ax2.axis('off')
ax2.set_title('sigma map', fontsize=15)
plt.colorbar(im2, ax=ax2)
plt.suptitle('Acquired noise map and estimated noise sigma', fontsize=15)
plt.show()



' T2 map estimation '

# echo times in ms
TE_time = np.array([40, 50, 60, 70, 100])
# data_array arranged in shape (80, 50, 120, 5)
data_decay = np.transpose(np.array([data_TE40, data_TE50, data_TE60, data_TE70, data_TE100]), (1, 2, 3, 0))

' T2-map calculation for a given data-set + mask '
WM_T2_map, WM_S0_fit = T2_map_calc(mask_wm, data_decay, TE_time, range=None)
GM_T2_map, GM_S0_fit = T2_map_calc(mask_gm, data_decay, TE_time, range=None)
T2_map, S0_fit = T2_map_calc(mask, data_decay, TE_time)

plt.imshow(T2_map[:, :, 60], cmap='gray', vmin=0, vmax=100)
plt.colorbar()
plt.title('$T_{2}$ map of a single slice')
plt.axis('off')
plt.show()

'  signal decay with TE '
wf_dur = np.arange(12, 22, 0.5)  # waveform durations

# parameters of the last measurement
TE_min = 40  # ms
dur = 12.5  # ms
spacing = 4.5  # ms
t_total = dur + spacing + dur

t_epi_div2 = TE_min - t_total  # t_total + T_epi/2 = TE, T_epi/2 represents a factor in ms


# partameters for new waveforms
TE_list = []  # TE for the new waveforms in ms

for i in range(len(wf_dur)):
    space = 4.5  # ms
    t_total_wf = wf_dur[i] + space + wf_dur[i]
    TE_list.append(t_total_wf + t_epi_div2)

print(TE_list)

WM_Sig_TE = S_TE(TE_list, WM_T2_map, WM_S0_fit, mask_wm)
GM_Sig_TE = S_TE(TE_list, GM_T2_map, GM_S0_fit, mask_gm)

print('wm sig te', WM_Sig_TE.shape)
print('sigma map', sigma_map.shape)


WM_snr_TE = Sig2Noise_TE(WM_Sig_TE, sigma_map, mask_wm)
GM_snr_TE = Sig2Noise_TE(GM_Sig_TE, sigma_map, mask_gm)

print('WM snr te', WM_snr_TE.shape)


' save the data sets '
#nib.Nifti1Image(WM_T2_map, affine).to_filename('Wm_T2map.nii')
#nib.Nifti1Image(GM_T2_map, affine).to_filename('Gm_T2map.nii')

#nib.Nifti1Image(WM_S0_fit, affine).to_filename('Wm_S0fitmap.nii')
#nib.Nifti1Image(GM_S0_fit, affine).to_filename('Gm_S0fitmap.nii')

#nib.Nifti1Image(WM_Sig_TE, affine).to_filename('Wm_sigTE.nii')
#nib.Nifti1Image(GM_Sig_TE, affine).to_filename('Gm_sigTE.nii')

#nib.Nifti1Image(WM_snr_TE, affine).to_filename('Wm_Snrmap_TE.nii')
#nib.Nifti1Image(GM_snr_TE, affine).to_filename('Gm_Snrmap_TE.nii')



' PLot SNR(TE)'

plt.figure(figsize=(10, 6))
plt.rc('font', size=15)
#plt.plot(TE_list, WM_snr_TE.mean(axis=(0, 1, 2)), label='WM SNR(TE)')
#plt.plot(TE_list, GM_snr_TE.mean(axis=(0, 1, 2)), label='GM SNR(TE)')
plt.plot(TE_list, np.true_divide(WM_snr_TE.sum(axis=(0, 1, 2)), (WM_snr_TE!=0).sum(axis=(0, 1, 2))) , label='WM SNR(TE)')
plt.plot(TE_list, np.true_divide(GM_snr_TE.sum(axis=(0, 1, 2)), (GM_snr_TE!=0).sum(axis=(0, 1, 2))) , label='GM SNR(TE)')
#plt.plot(TE_list, np.mean(WM_snr_TE[np.where(WM_snr_TE!=0)], axis=(0, 1, 2)) , label='WM SNR(TE)')
#plt.plot(TE_list, np.mean(GM_snr_TE[np.where(GM_snr_TE!=0)], axis=(0, 1, 2)) , label='GM SNR(TE)')
plt.xlabel('echo times [ms]')
plt.ylabel('SNR')
#plt.plot(TE_list[0], WMsnr_previous.mean(axis=(0, 1, 2)), 'ro', label='SNR of previous scan \n voxel [60, 40, 25] = {}'.format(snr_previous[60, 40, 25]))
plt.legend()
plt.title('SNR(TE)')
plt.show()

'Plot S(TE)'

plt.figure(figsize=(10, 6))
plt.rc('font', size=15)
#plt.plot(TE_list, np.mean(WM_Sig_TE[np.where(WM_Sig_TE!=0)], axis=(0, 1, 2)) , label='WM S(TE)')
#plt.plot(TE_list, np.mean(GM_Sig_TE[np.where(GM_Sig_TE!=0)], axis=(0, 1, 2)) , label='GM S(TE)')
plt.plot(TE_list, np.true_divide(WM_Sig_TE.sum(axis=(0, 1, 2)), (WM_Sig_TE!=0).sum(axis=(0, 1, 2))) , label='WM S(TE)')
plt.plot(TE_list, np.true_divide(GM_Sig_TE.sum(axis=(0, 1, 2)), (GM_Sig_TE!=0).sum(axis=(0, 1, 2))) , label='GM S(TE)')
plt.xlabel('echo times [ms]')
plt.ylabel('signal intensity')
plt.legend()
plt.title('S(TE)')
plt.show()



' load the estimated SNR maps for WM and GM '
' plot some histograms '

gm_snr_map, affine = load_data('Gm_Snrmap_TE01.nii')
wm_snr_map, affine = load_data('Wm_Snrmap_TE01.nii')

gm_te40 = gm_snr_map[:, :, :, 0]
gm_te50 = gm_snr_map[:, :, :, 1]
gm_te60 = gm_snr_map[:, :, :, 2]
gm_te70 = gm_snr_map[:, :, :, 3]
gm_te100 = gm_snr_map[:, :, :, 4]

wm_te40 = wm_snr_map[:, :, :, 0]
wm_te50 = wm_snr_map[:, :, :, 1]
wm_te60 = wm_snr_map[:, :, :, 2]
wm_te70 = wm_snr_map[:, :, :, 3]
wm_te100 = wm_snr_map[:, :, :, 4]
"""
print(np.max(gm_snr_map))
print(np.max(wm_snr_map))

plt.figure()
plt.hist(gm_te40[np.where(gm_te40!=0)], bins=100, alpha=0.5, label='TE 40 ms')
plt.hist(gm_te50[np.where(gm_te50!=0)], bins=100, alpha=0.5, label='TE 50 ms')
plt.hist(gm_te60[np.where(gm_te60!=0)], bins=100, alpha=0.5, label='TE 60 ms')
plt.hist(gm_te70[np.where(gm_te70!=0)], bins=100, alpha=0.5, label='TE 70 ms')
plt.hist(gm_te100[np.where(gm_te100!=0)], bins=100, alpha=0.5, label='TE 100 ms')
plt.title('Gray matter')
plt.legend()
plt.show()

plt.figure()
plt.hist(wm_te40[np.where(wm_te40!=0)], bins=100, alpha=0.5, label='TE 40 ms')
plt.hist(wm_te50[np.where(wm_te50!=0)], bins=100, alpha=0.5, label='TE 50 ms')
plt.hist(wm_te60[np.where(wm_te60!=0)], bins=100, alpha=0.5, label='TE 60 ms')
plt.hist(wm_te70[np.where(wm_te70!=0)], bins=100, alpha=0.5, label='TE 70 ms')
plt.hist(wm_te100[np.where(wm_te100!=0)], bins=100, alpha=0.5, label='TE 100 ms')
plt.title('White matter')
plt.legend()
plt.show()

plt.figure()
sns.distplot(gm_te40[np.where(gm_te40!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 40 ms')
sns.distplot(gm_te50[np.where(gm_te50!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 50 ms')
sns.distplot(gm_te60[np.where(gm_te60!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 60 ms')
sns.distplot(gm_te70[np.where(gm_te70!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 70 ms')
sns.distplot(gm_te100[np.where(gm_te100!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 100 ms')
plt.legend()
plt.title('Gray matter')
plt.show()

plt.figure()
sns.distplot(wm_te40[np.where(wm_te40!=0)], hist = True, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 40 ms')
sns.distplot(wm_te50[np.where(wm_te50!=0)], hist = True, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 50 ms')
sns.distplot(wm_te60[np.where(wm_te60!=0)], hist = True, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 60 ms')
sns.distplot(wm_te70[np.where(wm_te70!=0)], hist = True, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 70 ms')
sns.distplot(wm_te100[np.where(wm_te100!=0)], hist = True, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 100 ms')
plt.legend()
plt.title('White matter')
plt.show()
"""


fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharey=True)
fig.suptitle('SNR histogramms')

sns.distplot( gm_te40[np.where(gm_te40!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 40 ms', ax=axes[0])
sns.distplot(gm_te50[np.where(gm_te50!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 50 ms', ax=axes[0])
sns.distplot(gm_te60[np.where(gm_te60!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 60 ms', ax=axes[0])
sns.distplot(gm_te70[np.where(gm_te70!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 70 ms', ax=axes[0])
sns.distplot(gm_te100[np.where(gm_te100!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 100 ms', ax=axes[0])
axes[0].set_xlim(150, 650)
axes[0].set_title('Gray matter')
plt.legend()

sns.distplot(wm_te40[np.where(wm_te40!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 40 ms', ax=axes[1])
sns.distplot(wm_te50[np.where(wm_te50!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 50 ms', ax=axes[1])
sns.distplot(wm_te60[np.where(wm_te60!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 60 ms', ax=axes[1])
sns.distplot(wm_te70[np.where(wm_te70!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 70 ms', ax=axes[1])
sns.distplot(wm_te100[np.where(wm_te100!=0)], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'TE 100 ms', ax=axes[1])
axes[1].set_xlim(150, 650)
axes[1].set_title('White matter')
plt.legend()

plt.show()



