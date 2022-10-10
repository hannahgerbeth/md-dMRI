import numpy as np
import pylab as pl

from scipy.optimize import curve_fit#, check_grad


# spherical-mean cumulant + dot compartement model
def fitmom(bs, *coef):
	# E0 = (1-dotfrac)*exp(-bs*MD + 0.5*bs^2*MK) + dotfrac
	MD, MK, dotfrac = coef
	return (1-dotfrac)*np.exp(-bs*MD + 0.5*bs**2*MK) + dotfrac

# jacobian
def fitmom_jac(bs, *coef):
	MD, MK, dotfrac = coef
	tmp = np.exp(-bs*MD + 0.5*bs**2*MK)
	d_MD = -bs*(1-dotfrac)*tmp
	d_MK = 0.5*bs**2*(1-dotfrac)*tmp
	d_dotfrac = 1 - tmp
	return np.array([d_MD, d_MK, d_dotfrac]).T


""" toy example """


# bvals
bs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])

# S/S0 = 0.95*exp(-b*1  + 0.5*b^2*0.1) + 0.05
coef_gt = np.array([1.0, 0.1, 0.05])
signal_norm = fitmom(bs, *coef_gt)
pl.figure()
pl.plot(bs, signal_norm)
pl.show()


# init with mean mono exp on non-b0s with no dot
init = np.array([(np.log(signal_norm)/(-bs))[1:].mean(), 0, 0.0])
print(init)

result = curve_fit(fitmom, xdata=bs, ydata=signal_norm, p0=init, jac=fitmom_jac)

# print fit vs ground truth
print(result[0])
print(coef_gt)



# noisy example

# bvals
bs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6])

sigma = 0.03

# S/S0 = 0.95*exp(-b*1  + 0.5*b^2*0.1) + 0.05
coef = np.array([1.0, 0.2, 0.05])
signal_norm = fitmom(bs, *coef)
signal_norm_noisy = np.clip(signal_norm + sigma*np.random.randn(signal_norm.shape[0]), 0, 1)
pl.figure()
pl.plot(bs, signal_norm, label='noiseless', linewidth=2, alpha=0.75)
pl.plot(bs, signal_norm_noisy, label='noisy', linewidth=2, alpha=0.75)
pl.legend()
pl.show()



# init with mean mono exp on non-b0s with no dot
init = np.array([(np.log(signal_norm_noisy)/(-bs))[1:].mean(), 0, 0.0])


result_noisy = curve_fit(fitmom, xdata=bs, ydata=signal_norm_noisy, p0=init, jac=fitmom_jac)

# print fit vs ground truth
print(result_noisy[0])
print(coef)

pl.figure()
pl.plot(bs, signal_norm, label='noiseless', linewidth=2, alpha=0.75)
pl.plot(bs, signal_norm_noisy, label='noisy', linewidth=2, alpha=0.75)
pl.plot(bs, fitmom(bs, *result_noisy[0]), label='fit', linewidth=2, alpha=0.75)
pl.legend()
pl.show()









""" --- Generate very simple exponentially decaying data --- """
#DT1 = np.array(([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]))
#DT2 = np.array(([2., 0., 0.], [0., 2., 0.], [0., 0., 2.]))
#DTs = np.array((DT1, DT2))

import random
from Definitions import cov_mat, voigt_notation, inner_product
from Tensor_math_MPaquette import _S_ens
import matplotlib.pyplot as plt

N = 2
DT = np.zeros((N, 3, 3))
for i in range(DT.shape[0]):
	DT[i, 0,0] = random.random()*1e-4
	DT[i, 1,1] = random.random()*1e-4
	DT[i, 2,2] = DT[i, 1,1]

DT_mean = np.mean(voigt_notation(DT), axis=0)
C = cov_mat(voigt_notation(DT))

b_vals = np.arange(0, 10000, 1)

b_max = np.max(b_vals)
fake_bt = np.zeros((1000, 3, 3))
fake_bt[:, 0, 0] = 1
fake_bt[:, 1, 1] = 1
fake_bt[:, 2, 2] = 1
fake_bt *= np.linspace(0, b_max, 1000)[:,None,None]

S = np.clip(_S_ens(fake_bt, 1, DT_mean, C), 0,1)

plt.plot(S)
plt.show()


