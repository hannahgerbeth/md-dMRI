import numpy as np
import scipy
import scipy.stats
import scipy.optimize
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from dipy.data import get_sphere
from dipy.sims.voxel import all_tensor_evecs
from matplotlib import cm
from Tensor_math_MPaquette import tp
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere


# ----------------------------------------------------------------------------------------

"--- Mises Fisher Distribution---"
# use a mises fisher distribution to randomly chose the main directions of the tensors of the diffusion-tensor-distr.

def sphPDF(k, mu, direc):
    """
    :param k: concentration parameter
    :param mu: principle direction, e.g. [0,0,1]
    :param direc: directions on the sphere, e.g. sphere.vertices
    :return: pdf of the mises fisher distribution
    Generate the PDF for a von Mises-Fisher distribution p=3
    at locations direc for concentration k and mean orientation mu
    PDF = C3 * np.exp(k * mu.T * direc)
    """
    C3 = k / (2 * np.pi * (np.exp(k) - np.exp(-k)))
    tmp = np.exp(k * np.dot(direc, mu[:, None])).squeeze()
    return C3 * tmp


def DT_orientation(N, k, mu, threshold):
    """
    :param N: number of tensors
    :param k: concentration parameter of von Mises-Fisher distribution
    :param mu: principle direction of tensor orientations, e.g. [0,0,1]
    :threshold: what directions to choose based on the threshold
    :return: tensor orientations (N,3)
    """

    sphere = get_sphere('repulsion724').subdivide(1)

    'Use Mises Fischer Distribution to get the main directions of each tensor'
    mu /= np.linalg.norm(mu)
    # normalize the vector mu: divide vector by its norm (=Betrag)
    # normalzed mu = mu / sqrt(x^2 + y^2 + z^2)

    sph_dist = sphPDF(k, mu, sphere.vertices)
    sph_dist_peak = sph_dist.copy()  # Return an array copy of the given object

    random_dir_peak = np.zeros((N, 3))
    'Chose vectors from sphere'
    if threshold == 0:  # no threshold
        random_dir_idx_fullsphere = np.random.choice(sphere.vertices.shape[0], size=N, p=sph_dist / sph_dist.sum())
        #  randomly selects from the indices of the directional vectors (vertices) of the triangular sphere N
        #  and weights them with probability sph_dist = sphPDF(k, mu, sphere.vertices)
        random_dir_peak_new = sphere.vertices[random_dir_idx_fullsphere]
        for i in range(len(random_dir_peak_new)):
            random_dir_peak[i] = random_dir_peak_new[i]

    else:
        'Threshold'
        th = np.quantile(sph_dist_peak, threshold)  # some arbitrary threshold
        # Compute the q-th quantile of the data along the specified axis: Given a vector V of length N, the q-th quantile of V
        # is the value q of the way from the minimum to the maximum in a sorted copy of V.
        sph_dist_peak[sph_dist_peak < th] = 0  # setzt alle Werte/Wkt unterhalb th(threshold) gleich Null
        random_dir_idx_peak = np.random.choice(sphere.vertices.shape[0], size=N, p=sph_dist_peak / sph_dist_peak.sum())
        random_dir_peak_new = sphere.vertices[random_dir_idx_peak]
        for i in range(len(random_dir_peak_new)):
            random_dir_peak[i] = random_dir_peak_new[i]

    return random_dir_peak



"--- Diffusion-Tensor Eigenvectors---"

def DT_evecs(N, orientation):
    """
    eigenvectors calculated from the orientation of the tensor (which comes from the von Mises-Fisher distr.)
    :param N: number of tensors
    :param orientation: tensor orientations (N,3)
    :return: eigenvectors of the diffusion tensors (N, 3, 3)
    """

    evecs = np.zeros((N, 3, 3))
    tensor_orientations = orientation
    for i in range(N):
        # Dipy-function all_tensor_evecs: Given the principle tensor axis, return the array of all eigenvectors column-wise
        evecs[i] = all_tensor_evecs(tensor_orientations[i])
    return evecs


"--- Diffusion-Tensor Eigenvalues ---"
# calculate the eigenvalues from given (Mean Diffusivity)MD- and (Fractional anisotropy)FA-parameters
# each N_th tensor has 3 eigenvalues

# N ... number of tensors in the distribution
# D_shape ... shape of the diffusion tensors ('lin' = linear/ellipsoid, 'sph'=spherical)
# returns: an array (N, 3)

# generate a (N, ) array of random MD- and FA-values (chosen from a truncated gaussian)
def FA_gen(N, FA_mean, FA_sigma):
    """
    Generate a (N, ) array of random FA-values (chosen from a truncated gaussian)
    :param N: number of tensors
    :param FA_mean: give a mean FA-value
    :param FA_sigma: give a standard-deviation for the FA-values
    :return: FA-values (N, )
    """
    fa_lower = 0.001
    fa_upper = 0.99
    return scipy.stats.truncnorm.rvs(
        (fa_lower - FA_mean) / FA_sigma, (fa_upper - FA_mean) / FA_sigma, loc=FA_mean, scale=FA_sigma, size=N)

def MD_gen(N, MD_mean, MD_sigma):
    """
    Generate a (N, ) array of random MD-values (chosen from a truncated gaussian)
    :param N: number of tensors
    :param MD_mean: give a mean MD-value
    :param MD_sigma: give a standard-deviation for the MD-values
    :return: MD-values (N, )
    """
    md_lower = 0.0
    md_upper = np.inf
    return scipy.stats.truncnorm.rvs(
                (md_lower - MD_mean) / MD_sigma, (md_upper - MD_mean) / MD_sigma, loc=MD_mean, scale=MD_sigma, size=N)


def DT_evals(D_shape, MD, FA):
    """
    calculate the eigenvalues from given (fman diffusivity)MD- and (fractional anisotropy)FA-parameters, each N_th tensor has 3 eigenvalues
    :param D_shape: shape of the diffusion tensors ('lin' = linear/ellipsoid, 'sph'=spherical)
    :param MD: mean diffusivity (N, )
    :param FA: fractional anisotropy (N, )
    :return: eigenvalues of the diffusion tensors (N, 3)
    """

    # FA = sqrt((3*MD - 2*Lambda2 - Lambda2)^2 / ((3*MD - 2*Lambda2)^2 + 2*Lambda2^2))

    # 2 solution, the cigar D_shape (para>perp) and the pancake D_shape (perp>para)
    # some MD and FA combination will output imaginary solution for pancake mode

    if MD.shape == FA.shape:
        N = MD.shape[0]

    if D_shape == str('sph'):  # spherical tensor (para=perp)
        #  for spherical tensors: l_perp = l_para
        #  that means: fa = 0, md = (l_para + l_perp + l_perp)/3 = 3*lambda/3 = lambda
        evals = np.array((MD, MD, MD)).transpose()

    if D_shape == str('lin'):
        #  cigar-shape or linear (l_para > l_perp)
        sqrtroot = np.sqrt(3 * FA ** 2 * MD ** 2 - 2 * FA ** 4 * MD ** 2)

        l_perp1 = (2 * FA ** 2 * MD - sqrtroot - 3 * MD) / (2 * FA ** 2 - 3)  # first solution of 2. order equation
        l_perp2 = (2 * FA ** 2 * MD + sqrtroot - 3 * MD) / (2 * FA ** 2 - 3)  # second solution

        l_para1 = 3 * MD - 2 * l_perp1
        l_para2 = 3 * MD - 2 * l_perp2

        l_set1 = np.array((l_para1, l_perp1, l_perp1)).transpose()
        l_set2 = np.array((l_para2, l_perp2, l_perp2)).transpose()

        evals = np.zeros([N, 3])
        l1_bigger = np.where(l_set1[: , 0] >= l_set2[: , 0])
        l2_bigger = np.where(l_set2[: , 0] >= l_set1[: , 0])

        evals[l1_bigger[0]] = l_set1[l1_bigger[0]]
        evals[l2_bigger[0]] = l_set2[l2_bigger[0]]

        if len(np.where(evals < 0)) > 0:
            print('Negative Eigenvalues Detected')

        evals_smaller_zero = np.where(evals < 0)
        evals[evals_smaller_zero[0]] = np.zeros(len(l_set1[0]))

    return evals


"--- Diffusion Tensors ---"

def Diffusion_Tensors_manual(DT_evecs, DT_evals):
    """
    manually calculated diffusion tensors based on eigenvalue-decomposition
    taking the previously calculated evecs and evals in form of 3x3 matrices:
    E...Eigenvectors, L...eigenvalues as diagonal-matrix -> calculate D = E * L * E^(-1)
    :param DT_evecs: eigenvectors of the diffusion tensors (N, 3, 3)
    :param DT_evals: eigenvalues of the diffusion tensors (N, 3)
    :return: diffusion tensors (N, 3, 3)
    """

    N = DT_evecs.shape[0]

    D = np.zeros((N, 3, 3))
    E = DT_evecs  # evecs as a matrix
    L = np.array(DT_evals)  # take the evals
    if N == 1:
        L1 = np.diag(L[0:])  # put evals in a diagonal-matrix
        D[0] = np.dot(np.dot(E[0], L1), np.linalg.inv(E[0]))
    else:
        for i in range(N):
            L1 = np.diag(L[i])  # put evals in a diagonal-matrix
            D[i] = np.dot(np.dot(E[i], L1), np.linalg.inv(E[i]))

    return D


# ----------------------------------------------------------------------------------------
"--- B-Tensor Formalism ---"

"--- Voigt Notaion of a 3x3-Matrix ---"
def voigt_notation(T):
    """
    Given a tensor as a 3x3 matrix, rewrite the tensor in voigt notation, eq. 7 in Westin et al 2016
    d = ( dxx, dyy, dzz, sqrt(2)dyz, sqrt(2)dxz, sqrt(2)dxy)^T
    :param T: tensor-array (N,3,3)
    :return: tensor-array in voigt notation (N,6)
    """
    if T.shape[0] == 3:
        t = T[0][0], T[1][1], T[2][2], np.sqrt(2) * T[1][2], np.sqrt(2) * T[0][2], np.sqrt(2) * T[0][1]
    else:
        N = T.shape[0]
        t = np.zeros((N, 6))
        for i in range(N):
            # t[i] = T[i][0][0], T[i][1][1], T[i][2][2], T[i][1][2], T[i][0][2], T[i][0][1]
            t[i] = T[i][0][0], T[i][1][1], T[i][2][2], np.sqrt(2) * T[i][1][2], np.sqrt(2) * T[i][0][2], np.sqrt(2) * \
                   T[i][0][1]
    return t

# based on MPaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/tensor_math.py
def voigt_2_tensor(d):
    """
    Given a tensor in voigt notation, rewrite the tensor as a 3x3 matrix, eq. 7 in Westin et al 2016
    :param d: tensor-array in voigt notation (6,)
    :return: tensor-array (3,3)
    """
    D = np.zeros((3, 3))
    D[0, 0] = d[0]
    D[1, 1] = d[1]
    D[2, 2] = d[2]
    D[1, 2] = d[3] / np.sqrt(2)
    D[2, 1] = D[1, 2]
    D[0, 2] = d[4] / np.sqrt(2)
    D[2, 0] = D[0, 2]
    D[0, 1] = d[5] / np.sqrt(2)
    D[1, 0] = D[0, 1]
    return D

# from MPaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/tensor_math.py
def voigt_2_tensor_matlab_1d(D):
    """
    Diffusion tensor 3x3 to Voigt notation 1x6, eq. 7 in Westin et al 2016
	:param D: diffusion tensor (3,3)
	:return: diffusion tensor in voigt notation (6,)
    """
    d = np.zeros((6,))
    d[0] = D[0,0]
    d[1] = D[1,1]
    d[2] = D[2,2]
    d[3] = np.sqrt(2)*D[1,2]
    d[4] = np.sqrt(2)*D[0,2]
    d[5] = np.sqrt(2)*D[0,1]
    return d

# from MPaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/tensor_math.py
def c4_tensor_2_voigt(tmp):
    """
    C4 tensor 3x3x3x3 to Voigt notation 1x21
    :param tmp: C4 tensor (6,6)
    :return: C4 tensor in voigt notation (21,)
    """
    m = np.zeros((28))
    m[7] = tmp[0, 0]  # xx xx
    m[11] = tmp[0, 2]  # xx zz
    m[13] = tmp[0, 3]  # xx yz
    m[17] =tmp[0, 4]  # xx xz
    m[10] = tmp[0, 1]  # xx yy
    m[16] = tmp[0, 5]  # xx xy

    # tmp[1, 0] = C4[1, 1, 0, 0]
    m[8] = tmp[1, 1] # yy yy
    m[12] = tmp[1, 2]   # yy zz
    m[19] = tmp[1, 3]  # yy yz
    m[14] = tmp[1, 4]  # yy xz
    m[18] = tmp[1, 5]  # yy xy

    # tmp[2, 0] = C4[2, 2, 0, 0]
    # tmp[2, 1] = C4[2, 2, 1, 1]
    m[9] = tmp[2, 2]  # zz zz
    m[21] = tmp[2, 3]  # zz yz
    m[20] = tmp[2, 4]  # zz xz
    m[15] = tmp[2, 5]  # zz xy

    # tmp[3, 0] = C4[1, 2, 0, 0]
    # tmp[3, 1] = C4[1, 2, 1, 1]
    # tmp[3, 2] = C4[1, 2, 2, 2]
    m[24] = tmp[3, 3]  # yz yz
    # tmp[3, 4] = C4[1, 2, 0, 2]
    # tmp[3, 5] = C4[1, 2, 0, 1]

    # tmp[4, 0] = C4[0, 2, 0, 0]
    # tmp[4, 1] = C4[0, 2, 1, 1]
    # tmp[4, 2] = C4[0, 2, 2, 2]
    m[27] = tmp[4, 3]   # xz yz
    m[23] = tmp[4, 4]  # xz xz
    # tmp[4, 5] = C4[0, 2, 0, 1]

    # tmp[5, 0] = C4[0, 1, 0, 0]
    # tmp[5, 1] = C4[0, 1, 1, 1]
    # tmp[5, 2] = C4[0, 1, 2, 2]
    m[26] = tmp[5, 3]  # xy yz
    m[25] = tmp[5, 4]  # xy xz
    m[22] = tmp[5, 5]   # xy xy

    # symmetry time!
    # sym1: ab,cd = cd,ab
    tmp[1, 0] = tmp[0, 1]
    tmp[2, 0] = tmp[0, 2]
    tmp[2, 1] = tmp[1, 2]
    tmp[3, 0] = tmp[0, 3]
    tmp[3, 1] = tmp[1, 3]
    tmp[3, 2] = tmp[2, 3]
    tmp[3, 4] = tmp[4, 3]
    tmp[3, 5] = tmp[5, 3]
    tmp[4, 0] = tmp[0, 4]
    tmp[4, 1] = tmp[1, 4]
    tmp[4, 2] = tmp[2, 4]
    tmp[4, 5] = tmp[5, 4]
    tmp[5, 0] = tmp[0, 5]
    tmp[5, 1] = tmp[1, 5]
    tmp[5, 2] = tmp[2, 5]
    return m[7:]


"--- Inner Product = Frobenius-Skalarprodukt ---"
# inner product of two matrices A and B as mentioned in westing2016 (in analogon to the Frobenius-Skalarprodukt)
# returns: skalar

def inner_product(B, D):
    """
    Inner product of two matrices A and B as mentioned in westing2016 (in analogon to the Frobenius-Skalarprodukt)
    :param B: matrix (3,3), e.g. B-tensor
    :param D: matrix (3,3), e.g. D-tensor
    :return: skalar
    """
    # tensor product
    # ONE of the inputs can have extra dimension a the beginning
    return (B*D).sum((-1,-2))


"--- Signal generation functions ---"

def S(B, D):
    """
    Signal generation function: S(B) = exp(-<B,D>), Westin et al 2016
    :param B: matrix (3,3), B-tensor
    :param D: matrix (3,3), D-tensor
    :return: skalar
    """
    return np.exp(- inner_product(B, D))


def S_cum(B, S0, D):
    """
    Cumulant expansion of S for one B-Tensor with M directions and the averaged D-Tensor of N diffusion tensors
    S(B) = exp(-<B,<D>> + 1/2 <B°2,C >
    :param B: matrix (3,3), B-tensor
    :param S0: signal (skalar)
    :param D: matrix (3,3), D-tensor
    :return: skalar
    """

    # B = (3, 3) -> one b_value, one orientations-> original B-tensors are (N_b_tensors(=number of b_vals), N_b_orientations, 3, 3)
    # d = (6,)  (averaged) D-Distribution in Voigt Notation
    # c = (21,)

    d = voigt_notation(D).mean(axis=0)
    #print(d.shape)
    b = voigt_notation(B)
    c = cov_mat(voigt_notation(D))

    cum_signal = np.zeros((b.shape[0]))
    for i in range(b.shape[0]):
        # B°2 = b*b.T
        b2 = np.outer(b[i], b[i])

        # < B, < D >> + 0,5 * <B°2,C >
        # B°2 = b*b.T (b is B-Tensor in voigt-notation)
        # C... covariance-matrix
        cum_signal[i] = S0.mean() * np.exp(- np.matmul(b[i], d) + 0.5 * inner_product(b2, c))

    return cum_signal


def S_cum_ens(btensors, diff_tensors):
    """
    Cumulant expansion of S given an ensemble of N diffusion tensors and M various btensors
    use the cumulant expansion to calculate the signal
    :param: btensors: (M, 3, 3)
    :param: diff_tensors: (N, 3, 3)
    :return: signal: (M,)
    """
    # convert the diffusion tensors in voigt notation according to Westin2016
    DT_voigt = voigt_notation(diff_tensors)
    # calculate the mean diffusion tensor of the ensemble
    DT_voigt_mean = np.mean(DT_voigt, axis=0)
    # calculate the covariance matrix of the ensemble
    DT_covmat = cov_mat(DT_voigt)

    S_sim_cum = np.zeros(btensors.shape[0])
    for i in range(btensors.shape[0]):
        bt = np.asarray(voigt_notation(btensors[i]))
        S_sim_cum[i] = np.exp(- np.matmul(bt, DT_voigt_mean) + 0.5 * inner_product(np.outer(bt, bt), DT_covmat))
    return S_sim_cum


""""
def fit_signal(B, S0, d, c):
    # B = (3, 3) -> one b_value, one orientations-> original B-tensors are (N_b_tensors(=number of b_vals), N_b_orientations, 3, 3)

    b = voigt_notation(B)

    cum_signal = np.zeros((b.shape[0]))
    for i in range(b.shape[0]):
        # B°2 = b*b.T
        b2 = np.outer(b[i], b[i])
        # print(b2.shape)
        # < B, < D >> + 0,5 * <B°2,C >
        cum_signal[i] = S0.mean() * np.exp(- np.matmul(b[i], d) + 0.5 * inner_product(b2, c))

    return cum_signal
"""


def fit_signal_ens(btens, d_mean, covmat ):
    """
    From a given set of B-tensors, a mean D-tensor in Voigt-notation and a covariance matrix, calculate the cumulant approximation of the signal.
    Westin et al 2016, eq. 13; S \approx exp( - < B, mean(D) > + 0,5 * < B°2, C > )
    :param btens: (M, 3, 3)
    :param d_mean: (6,), average DT in Voigt-Notation
    :param covmat: (6, 6)
    """

    # temp1 = < B, mean(D) >
    temp1 = np.zeros(btens.shape[0])
    for i in range(btens.shape[0]):
        temp1[i] = inner_product(btens[i], voigt_2_tensor(d_mean))
        if np.isnan(temp1[i]) == True:
            print('Nan detected')

    # temp2 = < B°2, C >
    temp2 = np.zeros(btens.shape[0])
    for i in range(btens.shape[0]):
        b2 = np.outer(voigt_notation(btens[i]), voigt_notation(btens[i]))
        temp2[i] = inner_product(b2, covmat)
        if np.isnan(temp2[i]) == True:
            print('Nan detected')

    return np.exp( - temp1 + 0.5 * temp2)


def exp_signal(btens, d2):
    """
    Simple exponential signal approximation: S \approx exp( - < B, D > )
    given multiple B-tensors and one D-tensor in voigt-notation
    :param btens: (M, 3, 3)
    :param d2: (6,)
    :return: signal (M,)
    """
    temp = np.zeros(btens.shape[0] )
    for i in range(btens.shape[0]):
        temp[i] = np.exp(- tp(btens[i], voigt_2_tensor(d2)))
    return temp


# B = (N_orientations, 3, 3)
# D = (N_D-Tensors, 3, 3)
# S = < exp(- <B, D>) > : for each B-tensor in the array compute the inner produkt with every
# D-tensor from the D-Tensor-Distr. and average
def S_dis(B, D):
    """
    Simple exponential signal approximation for a set of D-tensors: S \approx exp( - < B, D > )
    :param B: (M, 3, 3)
    :param D: (N, 3, 3)
    :return: signal (M,)
    """
    signal = np.zeros((B.shape[0]))
    for i in range(B.shape[0]):
        #signal[i] = np.exp(- inner_product(B[i], D[:, :, :]).mean(axis=0))
        signal[i] = (np.exp(- inner_product(B[i], D[:, :, :])).mean(axis=0))
    return signal


def _S_simple(btensors, diff_tensors):
    """
    Simple exponential signal approximation for a set of D-tensors: S \approx exp( - < B, D > )
    :param btensors: (M, 3, 3)
    :param diff_tensors: (N, 3, 3)
    :return: signal (M,)
    """
    sig_simple = np.zeros((btensors.shape[0], diff_tensors.shape[0]))
    for i in range(btensors.shape[0]):
        for j in range(diff_tensors.shape[0]):
            sig_simple[i,j] = np.exp(- inner_product(btensors[i], diff_tensors[j]))
    return np.mean(sig_simple, axis=1)



"--- Noise Generation Function ---"

def noisy_signal(S, sigma):
    """
    Add noise with standard-deviation sigma to a given signal.
    Return either a Rician-distributed noisy signal (e.g. for low SNR) or a Gaussian-distributed noisy signal (e.g. for high SNR)
    :param S: signal (M,)
    :param sigma: standard-deviation (1,)
    :return: noisy signal (M,)
    """

    realchannel = np.random.normal(loc=0, scale=sigma, size=S.shape) + S
    imaginarychannel = np.random.normal(loc=0, scale=sigma, size=S.shape)

    riciannoise_signal = np.sqrt(realchannel**2 + imaginarychannel**2)
    gaussiannoise_signal = realchannel

    return riciannoise_signal, gaussiannoise_signal


def noise(S, sigma):
    """
    Generate pure noise with standard-deviation sigma. The noise is Gaussian distributed.
    :param S: signal of length (M,)
    :param sigma: Gaussian standard-deviation (1,)
    :return: noise (M,)
    """
    noise = sigma * np.random.randn(*S.shape)
    return noise


"--- Outer Product of two matrices in voigt notation ---"

def D_2_mean(d):
    """
    Calculate the outer product D°2 = d * d.T for each diffusion-tensor in voigt notation -> gives N times a 6x6 matrix
    <D°2> = <d d.T> = <di dj>, i,j = {xx, yy, zz, yz, xz, xy} (Westing2016)
    then calculate the average over all N matrices
    :param d: (N, 6)
    :return: (6, 6)
    """
    ddT = np.zeros((d.shape[0], 6, 6))  # stack of matrixes, 6x6 in x,y-plane, n along z-axis
    for i in range(d.shape[0]):
        ddT[i] = np.outer(d[i], d[i])
    return np.mean(ddT, axis=0)  # average over all matrices


def D_mean_2(d):
    """
    For a given set of diffusion-tensors in voigt notation, calculate the average tensor and then calculate the outer product
    <D>°2 = <d><d>.T = <di><dj>, i,j = {xx, yy, zz, yz, xz, xy} (Westing2016)
    :param d: (N, 6)
    :return: (6, 6)
    """
    d_mean = d.mean(axis=0)
    ddT = np.outer(d_mean, d_mean)
    return ddT


"--- Covariance Matrix of a matrix-distribution ---"

def cov_mat(d):
    """
    Calculate the covariance matrix of a DT-distribution in voigt notation as given in Westin2016: C = <D°2> - <D>°2
    :param d: (N, 6) Distribution of diffusion-tensors in voigt notation
    :return: (6, 6) Covariance matrix of the DTD
    """
    d1 = D_2_mean(d)
    d2 = D_mean_2(d)
    return d1 - d2

# version from MPaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/tensor_math.py - gives same output
def cov_mat_v2(d):
    """
    Calculate the covariance matrix of a DT-distribution in voigt notation as given in Westin2016: C = <D°2> - <D>°2
    :param d: (N, 6) Distribution of diffusion-tensors in voigt notation
    :return: (6, 6) Covariance matrix of the DTD
    """
    C = np.zeros((6, 6))
    def cij(i, j):
        return (d[:, i]*d[:, j]).mean() - d[:, i].mean()*d[:, j].mean()
    for i in range(6):
        for j in range(6):
            C[i, j] = cij(i, j)
    return C



"--- B-Tensor Eigenvalues ---"

def b_evals(b, B_shape):
    """
    Calculate the eigenvalues of a single b-tensor with a given shape-parameter: given a b-value and b_delta it calculates a b_tensor as in Topgaard2017
     b-Tensor = (b/3) * [1, 0, 0] + (b/3) * b_delta * [-1, 0, 0]
                        [0, 1, 0]                     [0, -1, 0]
                        [0, 0, 1]                     [0, 0, 2]
    :param b: b-value (1,)
    :param B_shape: a shape parameter (goes from -0.5(prolate) to 1.0(linear/ellipsoidal)) (1,)
    :return: eigenvalues of the b-tensor (3,3)
    """
    b_delta = B_shape
    b_tensor = (b / 3) * np.identity(3) + (b / 3) * b_delta * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    return np.flip(b_tensor)


"--- B-Tensor with a specific orientatoion ---"
# use the golden spiral method to generate N equally distant points (vectors) on a sphere
# calculate the all_tensor_evecs from the chosen orientation
# combine the previous eigenvalues with the eigenvectors to calculate a tensor with an orientation
# tensor calculation uses the eigenvalue-decomposition-algorithm:
#  E...Eigenvectors, L...eigenvalues as diagonal-matrix:  calculate D = E * L * E^(-1)

# N ... number of B-Tensors
# b ... b_value
# B-shape ... shape parameter (goes from -0.5(prolate) to 1.0(linear/ellipsoidal))

#  returns: any array with N b-tensors; they have the same eigenvalues, but N different orientations

def b_ten_orien(N):
    """
    - golden spiral method for semi-well distribution points -
    Use the golden spiral method to generate N equally distant points (vectors) on a sphere.
    Calculate the all_tensor_evecs from the chosen orientation
    Combine the previous eigenvalues with the eigenvectors to calculate a tensor with an orientation.
    Tensor calculation uses the eigenvalue-decomposition-algorithm: E...Eigenvectors, L...eigenvalues as diagonal-matrix
    Calculate D = E * L * E^(-1)
    :param N: number of B-tensor orientations
    :return: (N, 3) orientations
    """
    # Generating Equidistant Points on a Sphere
    # generate smooth transition orientation on the sphere
    golden_angle = np.pi * (3 - np.sqrt(5))  # = 2.39996 rad
    theta = golden_angle * np.arange(N)  # golden angle increment
    z = np.linspace(1 - 1.0 / N, 1.0 / N - 1, N)  # evenly spaced numbers vom -1 to 1, increment depends on n
    radius = np.sqrt(1 - z * z)  # radius at z

    points = np.zeros((N, 3))
    #  polar coordinates
    points[:, 0] = radius * np.cos(theta)  # x
    points[:, 1] = radius * np.sin(theta)  # y
    points[:, 2] = z  # z
    return points


def b_ten_orien_ESP(N):
    """ Dipy electro-static repulsion """
    theta = np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    sph = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))
    return sph.vertices

def b_tensors(N, b, b_ten_orien, B_shape): # N... number of b-ten orientations
    """
    Calculate N b-tensors with a given b-value, multiple orientations and a given shape-parameter.
    :param N: number of b-tensors
    :param b: b-value (1,)
    :param b_ten_orien: (N, 3) orientations
    :param B_shape: a shape parameter (goes from -0.5(prolate) to 1.0(linear/ellipsoidal)) (1,)
    :return: (N, 3, 3) b-tensors
    """
    B = np.zeros((N, 3, 3))
    b_evecs = np.zeros((N, 3, 3))

    for i in range(N):
        b_evecs[i] = all_tensor_evecs(b_ten_orien[i])
        L = b_evals(b, B_shape)  # take the evals that have the form of a diagonal matrix
        B[i] = np.dot(np.dot(b_evecs[i], L), np.linalg.inv(b_evecs[i]))
    return B

def build_btens(Nbt, bvals, orient, shape):
    """
    Build b-tensors with multiple b-values.
    :param Nbt: number of b-tensors
    :param bvals: (N,) b-values
    :param orient: (N, 3) orientations
    :param shape: a shape parameter (goes from -0.5(prolate) to 1.0(linear/ellipsoidal)) (1,)
    :return: (Nbt, 3, 3) b-tensors
    """
    bt_list = []
    for i in range(len(bvals)):
        bt_list.append(b_tensors(Nbt, bvals[i], orient, B_shape=shape))
    bt_arr = np.asarray(bt_list)
    bt = np.concatenate(bt_arr, axis=0)
    return bt

"--- A function to visualize a distribution of tensors ---"
# from MPaquette
def plot_tensors(b, fig, ax, factor):  # given an array b with many b_tensors
    """
    Plot a distribution of b-tensors. Implement as:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_tensors(D_tens, fig, ax, factor=10)
    plt.show()
    :param b: (N, 3, 3) b-tensors
    :param fig: figure
    :param ax: axis
    :param factor: scaling factor
    """

    # number of ellipsoids
    ellipNumber = len(b)

    # set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=ellipNumber)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # compute each and plot each ellipsoid iteratively
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')


    for k in range(ellipNumber):
        # your ellispsoid and center in matrix form
        A = b[k]
        #center = [k * np.random.random_sample(), k * np.random.random_sample(), k * np.random.random_sample()]
        center = [np.cbrt(ellipNumber) * np.random.rand(), np.cbrt(ellipNumber) * np.random.rand(), np.cbrt(ellipNumber) * np.random.rand()]
        #center = [np.random.rand(), np.random.rand(), np.random.rand()]

        # find the rotation matrix and radii of the axes
        #U, s, rotation = np.linalg.svd(A)
        #radii = 1.0 / np.sqrt(s) * 0.3  # reduce radii by factor 0.3
        eigval, eigvec = np.linalg.eig(A)
        radii = np.sqrt(eigval) * factor  #enlarge the radii by a factor so that the ellipsoids dont get too small
        rotation = np.array([eigvec[:, 0], eigvec[:, 1], eigvec[:, 2]])

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(k), linewidth=0.1, alpha=1, shade=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    return ax


def plot_tensors_v1(b):  # given an array b with many b_tensors
    # number of ellipsoids
    ellipNumber = len(b)

    # set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=ellipNumber)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # compute each and plot each ellipsoid iteratively
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in range(ellipNumber):
        # your ellispsoid and center in matrix form
        A = b[k]
        #center = [k * np.random.random_sample(), k * np.random.random_sample(), k * np.random.random_sample()]
        center = [np.cbrt(ellipNumber) * np.random.rand(), np.cbrt(ellipNumber) * np.random.rand(), np.cbrt(ellipNumber) * np.random.rand()]

        # find the rotation matrix and radii of the axes
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0 / np.sqrt(s) * 0.3  # reduce radii by factor 0.3

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(k), linewidth=0.1, alpha=1, shade=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()
    return fig




""" ---- Scalar invariants derived from QTI (Westin et al 2016) ---"""

def get_params(D):
    """
    Compute the scalar invariants of the diffusion tensor distribution, westin et al 2016
    :param D: (N, 3, 3) diffusion tensor distribution
    :return: C, V_md, C_md, C_mu, C_M, Cc, MD, FA_mu, FA, OP, MK, K_mu (scalar invariants)
    """
    E2iso = (1 / 3.) * np.eye(3)
    E4iso = (1 / 3.) * np.eye(6)
    Ebulk = np.zeros((6, 6))
    Ebulk[:3, :3] = 1 / 9.
    Eshear = E4iso - Ebulk

    d = voigt_notation(D)
    C = cov_mat(d)

    'Scalar Invariants derived from QTI (Westin2016)'
    # Mean Diffusivity
    # MD(<D>) = < <D>, Eiso > = 1/3 (d_xx + d_yy + d_zz)
    MD = (1/3.) * d.mean(axis=0)[:3].sum()

    # Variance in Mean-Diffusivities
    # can be interpreted as the buld or size variation of the diffusion tensors in the distribution
    V_md = inner_product(C, Ebulk)  # = V_bulk

    # Shear Variance
    V_shear = inner_product(C, Eshear)

    # total isotropic variance
    V_iso = inner_product(C, E4iso)

    ' Normalized scalar invariants derived from OTI (Westin2016)'
    # normalized size variance
    # C_md = 0 for when all microenvironments have the same size, C_md increases with increasing size variance
    C_md = V_md / inner_product(D_2_mean(d), Ebulk)

    # microscopic and macroscopic anisotropy
    C_mu = (3./2.) * (inner_product(D_2_mean(d), Eshear) / inner_product(D_2_mean(d), E4iso))
    FA_mu = np.sqrt(C_mu)  # micro FA

    # fractional anisotropy
    C_M = (3./2.) * (inner_product(D_mean_2(d), Eshear) / inner_product(D_mean_2(d), E4iso))
    FA = np.sqrt(C_M)

    # Microscopic orientation coherence (quantified by orientation parameter)
    OP_sq_1 = (inner_product(D_mean_2(d), Eshear) * inner_product(D_2_mean(d), Ebulk)) / \
              (inner_product(D_2_mean(d), Eshear) * inner_product(D_mean_2(d), Ebulk))
    OP_sq_2 = (inner_product(D_mean_2(d), Eshear)) / (inner_product(D_2_mean(d), Eshear))
    if np.isnan(OP_sq_1):
        OP_sq_1 = 0
    if np.isnan(OP_sq_2):
        OP_sq_2 = 0

    # OP = 0 (randomly orientated domain)
    # OP = 1 (perfectly coherent alignment)
    OP = np.sqrt((OP_sq_1+OP_sq_2)/2)
    #print(OP)

    # Westin-Order-Parameter for microscopic orientation coherence
    Cc = C_M/C_mu
    if np.isnan(Cc):
        Cc = 0

    ' Connections to DKI '
    # Kurtosis
    # Mean kurtosis = bulk kurtosis + shear kurtosis
    K_bulk = 3. * (inner_product(C, Ebulk))/(inner_product(D_mean_2(d), Ebulk))
    K_shear = (6./5.) * (inner_product(C, Eshear))/(inner_product(D_mean_2(d), Ebulk))
    MK = K_bulk + K_shear

    # microscopic kurtosis
    K_mu = (6./5.) * (inner_product(D_2_mean(d), Eshear))/(inner_product(D_mean_2(d), Ebulk))

    return C, V_md, C_md, C_mu, C_M, Cc, MD, FA_mu, FA, OP, MK, K_mu


# from MPaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/tensor_math.py
def get_metric_michael(D):
    """
    Compute the scalar invariants of the diffusion tensor distribution, westin et al 2016
    :param D: (N, 3, 3) diffusion tensor distribution
    :return: C, Cmd, Cmu, CM, Cc, MD, mu_FA, FA (selected scalar invariants)
    """

    E2iso = (1 / 3.) * np.eye(3)
    E4iso = (1 / 3.) * np.eye(6)
    Ebulk = np.zeros((6, 6))
    Ebulk[:3, :3] = 1 / 9.
    Eshear = E4iso - Ebulk

    d = voigt_notation(D)
    C = cov_mat(d) # add w
    D2E = D_2_mean(d) # add w
    DE2 = D_mean_2(d) # add w

    MD = (1 / 3.) * d.mean(axis=0)[:3].sum()

    Vbulk = inner_product(C, Ebulk) # Vmd
    # Vshear = tp(C, Eshear)
    # Viso = tp(C, E4iso) # Vmd + Vshear
    Cmd = Vbulk / inner_product(D2E, Ebulk)
    Cmu = (3/2.)*inner_product(D2E, Eshear) / inner_product(D2E, E4iso)
    mu_FA = np.sqrt(Cmu)

    CM = (3/2.)*inner_product(DE2, Eshear) / inner_product(DE2, E4iso)
    FA = np.sqrt(CM)
    Cc = CM / Cmu

    return C, Cmd, Cmu, CM, Cc, MD, mu_FA, FA


""" --- DTD Fitting Function --- """

# from Mpaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/dtd_cov.py
# updated version
def dtd_cov_1d_data2fit_v1(S, bt, cond_limit=1e-10, clip_eps=1e-16):
    """
    Fit the DTD-model to the data, westin et al 2016
    :param S: (N,) signal (unnormalized or normalized)
    :param bt: (N, 3, 3) b-tensors (in whatever unit, we will keep them)
    :param cond_limit: condition limit for the linear system
    :param clip_eps: clipping value
    :return: result (28,) (solution vector of the linear least squares fit)
    """

    # clipping signal
    S = np.clip(S, clip_eps, np.inf)

    # number of B-tensor
    N = S.shape[0]


    # setting up system matrix identically to md-dmri toolbox: dtd_cov_1d_data2fit.m and tm_1x6_to_1x21.m
    # NOTE: this ordering is different then the one from Westin2016 paper
    # for experiments with b-tensors of rank 2 or 3, b4 has major symmetry and thus 21 unique elements (westin2016)

    # set up a system that looks like that (westin2016):
    # (log S1) = ( 1  -b2_1.T  0.5*b4_1.T)
    # (log S2) = ( 1  -b2_2.T  0.5*b4_2.T) * (S0  <d>  c)
    # ( ...  ) = (...    ...        ...  )
    # (log SN) = ( 1  -b2_N.T  0.5*b4_N.T)
    # <d> ... average of D-Tensor-Distribution as a vector (in voigt notation (other notation order than used before))
    # see: Twenty-five Pitfalls in the Analysis of Diffusion MRI Data, Derek K. Jones and Mara Cercignani, DOI:10.1002/nbm.1543
    # c ... covariance-matrix as a (21x1) vector
    b0 = np.ones((N, 1))
    b2 = np.zeros((N, 6))
    b4 = np.zeros((N, 21))

    sqrt2 = np.sqrt(2)
    b2[:, 0] = bt[:, 0, 0]  # xx
    b2[:, 1] = bt[:, 1, 1]  # yy
    b2[:, 2] = bt[:, 2, 2]  # zz
    b2[:, 3] = bt[:, 0, 1] * sqrt2  # xy
    b2[:, 4] = bt[:, 0, 2] * sqrt2  # xz
    b2[:, 5] = bt[:, 1, 2] * sqrt2  # yz

    b4[:, 0] = bt[:, 0, 0] * bt[:, 0, 0]  # xx xx
    b4[:, 1] = bt[:, 1, 1] * bt[:, 1, 1]  # yy yy
    b4[:, 2] = bt[:, 2, 2] * bt[:, 2, 2]  # zz zz
    b4[:, 3] = bt[:, 0, 0] * bt[:, 1, 1] * sqrt2  # xx yy
    b4[:, 4] = bt[:, 0, 0] * bt[:, 2, 2] * sqrt2  # xx zz
    b4[:, 5] = bt[:, 1, 1] * bt[:, 2, 2] * sqrt2  # yy zz
    b4[:, 6] = bt[:, 0, 0] * bt[:, 1, 2] * 2  # xx yz
    b4[:, 7] = bt[:, 1, 1] * bt[:, 0, 2] * 2  # yy xz
    b4[:, 8] = bt[:, 2, 2] * bt[:, 0, 1] * 2  # zz xy
    b4[:, 9] = bt[:, 0, 0] * bt[:, 0, 1] * 2  # xx xy
    b4[:, 10] = bt[:, 0, 0] * bt[:, 0, 2] * 2  # xx xz
    b4[:, 11] = bt[:, 1, 1] * bt[:, 0, 1] * 2  # yy xy
    b4[:, 12] = bt[:, 1, 1] * bt[:, 1, 2] * 2  # yy yz
    b4[:, 13] = bt[:, 2, 2] * bt[:, 0, 2] * 2  # zz xz
    b4[:, 14] = bt[:, 2, 2] * bt[:, 1, 2] * 2  # zz yz
    b4[:, 15] = bt[:, 0, 1] * bt[:, 0, 1] * 2  # xy xy
    b4[:, 16] = bt[:, 0, 2] * bt[:, 0, 2] * 2  # xz xz
    b4[:, 17] = bt[:, 1, 2] * bt[:, 1, 2] * 2  # yz yz
    b4[:, 18] = bt[:, 0, 1] * bt[:, 0, 2] * 2 * sqrt2  # xy xz
    b4[:, 19] = bt[:, 0, 1] * bt[:, 1, 2] * 2 * sqrt2  # xy yz
    b4[:, 20] = bt[:, 0, 2] * bt[:, 1, 2] * 2 * sqrt2  # xz yz

    # setting up the (N,28) system matrix
    # np.concatenate: Join a sequence of arrays along an existing axis
    X = np.concatenate((b0, -b2, 0.5 * b4), axis=1)

    # computing the heteroscedasticity correction matrix H
    # H is a diagonal matrix with the signal amplitudes as diagonal elements
    # H = np.diag(S)
    H = np.diag(S ** 2)

    # check the condition number
    # we want to compute the pseudoinvariance: beta* = (X.T * X)^-1 * X.T * S (westin2016, eq 46)
    # therefore, X.T * X must have full rank: rank(X.T*X) = 28 (=1+6+21)
    # computing X.T * H * X
    tmp = np.dot(np.dot(X.T, H), X)
    # computing Matlab's rcond (Reciprocal condition number) equivalent
    # Reciprocal condition number, returned as a scalar. The data type of C (cond. number) is the same as A (quadratic matrix).
    # The reciprocal condition number is a scale-invariant measure of how close a given matrix is to the set of singular matrices.
    # - If C is near 0, the matrix is nearly singular and badly conditioned.
    # - If C is near 1.0, the matrix is well conditioned.
    rep_cond = np.linalg.cond(tmp) ** -1
    if rep_cond < cond_limit:
        #print('rcond fail in dtd_covariance_1d_data2fit {}'.format(rep_cond))
        return np.zeros(28)

    # pseudoinverse modelfit (similar to Matlab's 'backslash')
    # "A x = b"
    # [H * X] * m = [H * ln(S)]
    # Xh * m = log_Sh
    log_Sh = np.dot(H, np.real(np.log(S)))
    Xh = np.dot(H, X)
    m = np.linalg.lstsq(Xh, log_Sh, rcond=None)

    # probably need to be compatible with the rest
    # m[0] = np.exp(m[0])

    return m[0]


""" --- Mean squared error of estimated covariance matrix --- """

def err_covmat(C_dis, C_fit):
    # rmse = sqrt ( 1/n * Sum(i to n) (x_obs,i - x_model,i)^2 ) root mean squared error
    rmse = np.sqrt( (1 / (C_dis.shape[0] * C_dis.shape[1])) * np.sum((C_dis - C_fit) ** 2))
    # nrmse = rmse / mean(x_obs)
    nrmse = rmse / (C_dis.mean(axis=0)).mean(axis=0)
    # percentage
    #perc_err = np.abs(np.linalg.norm(C_dis) - np.linalg.norm(C_fit))/np.linalg.norm(C_dis) *100
    perc_err = np.abs(np.linalg.norm(C_dis - C_fit)) / np.linalg.norm(C_dis) * 100
    return rmse, nrmse, perc_err

