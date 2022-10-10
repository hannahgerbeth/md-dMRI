import numpy as np


def tp(B, D):
	# tensor product
	# ONE of the inputs can have extra dimension a the beginning
	return (B*D).sum((-1,-2))


# signal generation simple
def _S(B, D):
	return np.exp(-tp(B,D))


# signal generation from cummulant approximation
def _S_ens(B, s0, d, c):
	# S(B) \approx S0 * exp(- < B, <D> >  +  1/2  < BxB, C >)
	# B is (N,3,3)
	# d is (6,)
	# c is (21,)
	# voigt the b tensor
	bt = voigt_tensor(B)
	# get BxB in 6x6 form
	bt2 = np.array([np.outer(bt[i],bt[i]) for i in range(B.shape[0])])
	# < B, <D> >
	cum2 = np.dot(bt, d)
	# < BxB, C >
	cum4 = tp(bt2,c)
	# return signal
	return s0 * np.exp(-cum2 + 0.5*cum4)


# signal generation from tensor distribution
def dist_S(B, D, PD, S0=1):
	# B is (3,3)
	# D is (N,3,3)
	# PD is (N,)
	return S0*(PD*_S(B, D)).sum(axis=0)


# Nx3x3 to Nx6
# with ordering [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
def voigt_tensor(D):
	# Nx3x3 to Nx6
	# eq. 7 in Westin et al 2016
	d = np.zeros((D.shape[0], 6))
	d[:,0] = D[:,0,0]
	d[:,1] = D[:,1,1]
	d[:,2] = D[:,2,2]
	d[:,3] = np.sqrt(2)*D[:,1,2]
	d[:,4] = np.sqrt(2)*D[:,0,2]
	d[:,5] = np.sqrt(2)*D[:,0,1]
	return d


def d_to_D(d):
	# assumes d is ordered:
	# xx yy zz yz xz xy
	# with "multiplier"
	# 1 1 1 sqrt(2) sqrt(2) sqrt(2)
	D = np.zeros((3,3))
	D[0,0] = d[0]
	D[1,1] = d[1]
	D[2,2] = d[2]
	D[1,2] = d[3] / np.sqrt(2)
	D[2,1] = D[1,2]
	D[0,2] = d[4] / np.sqrt(2)
	D[2,0] = D[0,2]
	D[0,1] = d[5] / np.sqrt(2)
	D[1,0] = D[0,1]

	return D


def ensD_2(d):
	# <D> xo 2
	# from Voigt notation
	# eq. 9 in Westin et al 2016
	tmp = d.mean(axis=0)
	return np.outer(tmp, tmp)


def w_ensD_2(d, P):
	# <D> xo 2
	# from Voigt notation
	# eq. 9 in Westin et al 2016
	tmp = (P[:, None, None]*d).sum(axis=0)
	return np.outer(tmp, tmp)


def ens_D2(d):
	# <D xo 2>
	# from Voigt notation
	# eq. 8 in Westin et al 2016
	tmp = np.zeros((d.shape[0], 6, 6))
	for i in range(d.shape[0]):
		tmp[i] = np.outer(d[i],d[i])
	return tmp.mean(axis=0)


def w_ens_D2(d, P):
	# <D xo 2>
	# from Voigt notation
	# eq. 8 in Westin et al 2016
	tmp = np.zeros((d.shape[0], 6, 6))
	for i in range(d.shape[0]):
		tmp[i] = np.outer(d[i],d[i])
	return (P[:, None, None]*tmp).sum(axis=0)


# TODO add weight
def cov_voigt(d):
	# Covariance (in 6x6 form) of the distribution define by the Nx6 vector of Voigt notation tensor
	# eq. 10 and 11 in Westin et al 2016
	# ind = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
	C = np.zeros((6,6))
	def Cij(i,j):
		return (d[:,i]*d[:,j]).mean() - d[:,i].mean()*d[:,j].mean()
	for i in range(6):
		for j in range(6):
			C[i,j] = Cij(i,j)
	return C



# 3x3x3x3 to 6x6
def mandel4(C4):
	tmp = np.zeros((6,6))
	tmp[0, 0] = C4[0, 0, 0, 0]
	tmp[0, 1] = C4[0, 0, 1, 1]
	tmp[0, 2] = C4[0, 0, 2, 2]
	tmp[0, 3] = C4[0, 0, 1, 2]*np.sqrt(2)
	tmp[0, 4] = C4[0, 0, 0, 2]*np.sqrt(2)
	tmp[0, 5] = C4[0, 0, 0, 1]*np.sqrt(2)

	tmp[1, 0] = C4[1, 1, 0, 0]
	tmp[1, 1] = C4[1, 1, 1, 1]
	tmp[1, 2] = C4[1, 1, 2, 2]
	tmp[1, 3] = C4[1, 1, 1, 2]*np.sqrt(2)
	tmp[1, 4] = C4[1, 1, 0, 2]*np.sqrt(2)
	tmp[1, 5] = C4[1, 1, 0, 1]*np.sqrt(2)

	tmp[2, 0] = C4[2, 2, 0, 0]
	tmp[2, 1] = C4[2, 2, 1, 1]
	tmp[2, 2] = C4[2, 2, 2, 2]
	tmp[2, 3] = C4[2, 2, 1, 2]*np.sqrt(2)
	tmp[2, 4] = C4[2, 2, 0, 2]*np.sqrt(2)
	tmp[2, 5] = C4[2, 2, 0, 1]*np.sqrt(2)

	tmp[3, 0] = C4[1, 2, 0, 0]*np.sqrt(2)
	tmp[3, 1] = C4[1, 2, 1, 1]*np.sqrt(2)
	tmp[3, 2] = C4[1, 2, 2, 2]*np.sqrt(2)
	tmp[3, 3] = C4[1, 2, 1, 2]*2
	tmp[3, 4] = C4[1, 2, 0, 2]*2
	tmp[3, 5] = C4[1, 2, 0, 1]*2

	tmp[4, 0] = C4[0, 2, 0, 0]*np.sqrt(2)
	tmp[4, 1] = C4[0, 2, 1, 1]*np.sqrt(2)
	tmp[4, 2] = C4[0, 2, 2, 2]*np.sqrt(2)
	tmp[4, 3] = C4[0, 2, 1, 2]*2
	tmp[4, 4] = C4[0, 2, 0, 2]*2
	tmp[4, 5] = C4[0, 2, 0, 1]*2

	tmp[5, 0] = C4[0, 1, 0, 0]*np.sqrt(2)
	tmp[5, 1] = C4[0, 1, 1, 1]*np.sqrt(2)
	tmp[5, 2] = C4[0, 1, 2, 2]*np.sqrt(2)
	tmp[5, 3] = C4[0, 1, 1, 2]*2
	tmp[5, 4] = C4[0, 1, 0, 2]*2
	tmp[5, 5] = C4[0, 1, 0, 1]*2

	return tmp


# kernel definition for the Cov tensor operation
E2iso = (1/3.)*np.eye(3)
E4iso = (1/3.)*np.eye(6)
Ebulk = np.zeros((6,6))
Ebulk[:3,:3] = 1/9.
Eshear = E4iso - Ebulk


# ground truth variances from tensor distribution
def get_metric(D, P=None):
	# D is Nx3x3 tensor
	# P is he distribution weight over D

	# uniform
	if P is None:
		P = np.ones(D.shape[0])
	# Normalized
	P /= P.sum()

	d = voigt_tensor(D)
	C = cov_voigt(d) # add w
	D2E = ens_D2(d) # add w
	DE2 = ensD_2(d) # add w

	MD = (1 / 3.) * d.mean(axis=0)[:3].sum()

	Vbulk = tp(C, Ebulk) # Vmd
	# Vshear = tp(C, Eshear)
	# Viso = tp(C, E4iso) # Vmd + Vshear
	Cmd = Vbulk / tp(D2E, Ebulk)

	Cmu = (3/2.)*tp(D2E, Eshear) / tp(D2E, E4iso)
	CM = (3/2.)*tp(DE2, Eshear) / tp(DE2, E4iso)
	FA = np.sqrt(CM)
	Cc = CM / Cmu

	return Cmd, Cmu, CM, Cc, MD, FA