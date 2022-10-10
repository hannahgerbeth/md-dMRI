import numpy as np

# from vector_math import
from Tensor_math_MPaquette import tp

# from Mpaquette: https://github.com/mpaquette/gnlc_waveform/blob/master/dtd_cov.py

def dtd_cov_1d_data2fit(S, bt, cond_limit=1e-10, clip_eps=1e-16):
    # S is a (N,) signal (unnormalized)
    # bt is (N,3,3) full B-tensor (in whatever unit, we will keep them)

    # clipping signal
    S = np.clip(S, clip_eps, np.inf)

    # number of B-tensor
    N = S.shape[0]

    # setting up system matrix identically to md-dmri toolbox: dtd_cov_1d_data2fit.m and tm_1x6_to_1x21.m
    # NOTE: this ordering is different then the one from Westin2016 paper
    b0 = np.ones((N,1))
    b2 = np.zeros((N,6))
    b4 = np.zeros((N,21))

    sqrt2 = np.sqrt(2)
    b2[:,0] = bt[:,0,0] 	  # xx
    b2[:,1] = bt[:,1,1] 	  # yy
    b2[:,2] = bt[:,2,2] 	  # zz
    b2[:,3] = bt[:,0,1]*sqrt2 # xy
    b2[:,4] = bt[:,0,2]*sqrt2 # xz
    b2[:,5] = bt[:,1,2]*sqrt2 # yz

    b4[:,0]  = bt[:,0,0]*bt[:,0,0]         # xx xx
    b4[:,1]  = bt[:,1,1]*bt[:,1,1]         # yy yy
    b4[:,2]  = bt[:,2,2]*bt[:,2,2]         # zz zz
    b4[:,3]  = bt[:,0,0]*bt[:,1,1]*sqrt2   # xx yy
    b4[:,4]  = bt[:,0,0]*bt[:,2,2]*sqrt2   # xx zz
    b4[:,5]  = bt[:,1,1]*bt[:,2,2]*sqrt2   # yy zz
    b4[:,6]  = bt[:,0,0]*bt[:,1,2]*2       # xx yz
    b4[:,7]  = bt[:,1,1]*bt[:,0,2]*2       # yy xz
    b4[:,8]  = bt[:,2,2]*bt[:,0,1]*2       # zz xy
    b4[:,9]  = bt[:,0,0]*bt[:,0,1]*2       # xx xy
    b4[:,10] = bt[:,0,0]*bt[:,0,2]*2       # xx xz
    b4[:,11] = bt[:,1,1]*bt[:,0,1]*2       # yy xy
    b4[:,12] = bt[:,1,1]*bt[:,1,2]*2       # yy yz
    b4[:,13] = bt[:,2,2]*bt[:,0,2]*2       # zz xz
    b4[:,14] = bt[:,2,2]*bt[:,1,2]*2       # zz yz
    b4[:,15] = bt[:,0,1]*bt[:,0,1]*2       # xy xy
    b4[:,16] = bt[:,0,2]*bt[:,0,2]*2       # xz xz
    b4[:,17] = bt[:,1,2]*bt[:,1,2]*2       # yz yz
    b4[:,18] = bt[:,0,1]*bt[:,0,2]*2*sqrt2 # xy xz
    b4[:,19] = bt[:,0,1]*bt[:,1,2]*2*sqrt2 # xy yz
    b4[:,20] = bt[:,0,2]*bt[:,1,2]*2*sqrt2 # xz yz


    # setting up the (N,28) system matrix
    X = np.concatenate((b0, -b2, 0.5*b4), axis=1)

    # computing the heteroscedasticity correction matrix
    C2 = np.diag(S**2)

    # check the condition number
    # computing X' * C2 * X
    tmp = np.dot(np.dot(X.T, C2), X)
    # computing Matlab's rcond equivalent
    rep_cond = np.linalg.cond(tmp)**-1
    if rep_cond < cond_limit:
        print('rcond fail in dtd_covariance_1d_data2fit {}'.format(rep_cond))
        return np.zeros(28)

    # pseudoinverse modelfit (similar to Matlab's 'backslash')
    # "A x = b"
    # [C2 * X] * m = [C2 * ln(S)]
    m = np.linalg.lstsq(np.dot(C2,X), np.dot(C2,np.real(np.log(S))), rcond=None)

    # probably need to be compatible with the rest
    # m[0] = np.exp(m[0])
    return m[0]


# convert the output of the fit back to d and c "format"
def convert_m(m):
    s0 = np.exp(m[0])

    d2 = np.zeros(6)
    d2[0] = m[1] # xx
    d2[1] = m[2] # yy
    d2[2] = m[3] # zz
    d2[3] = m[4] # yz
    d2[4] = m[5] # xz
    d2[5] = m[6] # xy

    # d4
    tmp = np.zeros((6,6))
    tmp[0, 0] = m[7]  # xx xx
    tmp[0, 1] = m[10] # xx yy
    tmp[0, 2] = m[11] # xx zz
    tmp[0, 3] = m[13] # xx yz
    tmp[0, 4] = m[17] # xx xz
    tmp[0, 5] = m[16] # xx xy

    # tmp[1, 0] = C4[1, 1, 0, 0]
    tmp[1, 1] = m[8]  # yy yy
    tmp[1, 2] = m[12] # yy zz
    tmp[1, 3] = m[19] # yy yz
    tmp[1, 4] = m[14] # yy xz
    tmp[1, 5] = m[18] # yy xy

    # tmp[2, 0] = C4[2, 2, 0, 0]
    # tmp[2, 1] = C4[2, 2, 1, 1]
    tmp[2, 2] = m[9]  # zz zz
    tmp[2, 3] = m[21] # zz yz
    tmp[2, 4] = m[20] # zz xz
    tmp[2, 5] = m[15] # zz xy

    # tmp[3, 0] = C4[1, 2, 0, 0]
    # tmp[3, 1] = C4[1, 2, 1, 1]
    # tmp[3, 2] = C4[1, 2, 2, 2]
    tmp[3, 3] = m[24] # yz yz
    # tmp[3, 4] = C4[1, 2, 0, 2]
    # tmp[3, 5] = C4[1, 2, 0, 1]

    # tmp[4, 0] = C4[0, 2, 0, 0]
    # tmp[4, 1] = C4[0, 2, 1, 1]
    # tmp[4, 2] = C4[0, 2, 2, 2]
    tmp[4, 3] = m[27] # xz yz
    tmp[4, 4] = m[23] # xz xz
    # tmp[4, 5] = C4[0, 2, 0, 1]

    # tmp[5, 0] = C4[0, 1, 0, 0]
    # tmp[5, 1] = C4[0, 1, 1, 1]
    # tmp[5, 2] = C4[0, 1, 2, 2]
    tmp[5, 3] = m[26] # xy yz
    tmp[5, 4] = m[25] # xy xz
    tmp[5, 5] = m[22] # xy xy

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

    # S0, <D>, Cov
    return s0, d2, tmp


# kernel definition for the Cov tensor operation
E2iso = (1/3.)*np.eye(3)
E4iso = (1/3.)*np.eye(6)
Ebulk = np.zeros((6,6))
Ebulk[:3,:3] = 1/9.
Eshear = E4iso - Ebulk


# computes all the metrics from the fit
def decode_m(d2, c4, reg=1e-4):

    # copied from tm_dt_to_dps
    # compute the second moments of the mean diffusion tensor
    d4 = np.outer(d2,d2)

    # fourth order tensor operators needed for the calculations
    V_MD2    = np.clip(tp(d4, Ebulk), 0, np.inf)
    V_iso2   = np.clip(tp(d4, E4iso), 0, np.inf)
    V_shear2 = np.clip(tp(d4, Eshear), 0, np.inf)

    # DTI parameters derived from the diffusion tensor
    MD     = d2[:3].mean()
    FA     = np.sqrt(0.5*(3 - ((3*max(MD,reg))**2 / (d2**2).sum())))

    # # Compute parameters using eigenvalues L and primary direction U
    # [L,U]  = tm_1x6_eigvals(dt_1x6);
    # ad     = (real(L(:,1)), 1);
    # rd     = (mean(real(L(:,2:3)),2), 1);
    # u      = (U,3);


    # copied from tmp_ct_to_dps.m
    # DTD paramters derived from first and second cumulant term
    V_MD = np.clip(tp(c4, Ebulk), 0, np.inf)
    V_iso = np.clip(tp(c4, E4iso), 0, np.inf)

    V_MD1 = V_MD + V_MD2
    V_iso1 = V_iso + V_iso2

    V_shear = np.clip(tp(c4, Eshear), 0, np.inf)
    V_shear1 = V_shear + V_shear2

    # ********* Normalized variance measures ************
    C_MD = V_MD / max(V_MD1, reg)
    C_mu = 1.5 * V_shear1 / max(V_iso1, reg)
    C_M = 1.5 * V_shear2 / max(V_iso2, reg)
    C_c = C_M / max(C_mu, reg)

    # ********* Kurtosis measures ************
    # Naming these according to the dtd_gamma nomenclature
    MKi = 3 * V_MD / max(V_MD2, reg)
    MKa = (6/5.) * V_shear1 / max(V_MD2, reg) # K_micro in Westin16
    MKt = MKi + MKa
    MKad = (6/5.) * V_shear / max(V_MD2, reg) # anisotropy and dispersion
    MK = MKad + MKi # conventional kurtosis
    MKd = MKa - MKad # conventional kurtosis
    uFA = np.sqrt(C_mu)

    S_I = np.sqrt(V_MD * (V_MD > 0))
    S_A = np.sqrt(V_shear1 * (V_shear1 > 0))

    return V_MD2,V_iso2,V_shear2,MD,FA,V_MD,V_iso,V_MD1,V_iso1,V_shear,V_shear1,C_MD,C_mu,C_M,C_c,MKi,MKa,MKt,MKad,MK,MKd,uFA,S_I,S_A


def decode_m_v2(d2, c4, reg=1e-10):

    # copied from tm_dt_to_dps
    # compute the second moments of the mean diffusion tensor
    d4 = np.outer(d2,d2)  # d4 = < D > ° 2

    # fourth order tensor operators needed for the calculations
    # VMD = < c4, Ebulk > -> c4 = <D°2> (unknown) - <D>°2 (from fit)
    # VMD = VMD1 - VMD2 = < <D°2>, Ebilk > -  < <D>°2, Ebulk >
    # same applies for Viso, Vshear
    V_MD2    = np.clip(tp(d4, Ebulk), 0, np.inf) # VMD2 = < <D>°2, Ebulk >
    V_iso2   = np.clip(tp(d4, E4iso), 0, np.inf)
    V_shear2 = np.clip(tp(d4, Eshear), 0, np.inf)

    # DTI parameters derived from the diffusion tensor
    MD     = d2[:3].mean()
    FA     = np.sqrt(0.5*(3 - ((3*max(MD,reg))**2 / (d2**2).sum())))

    # # Compute parameters using eigenvalues L and primary direction U
    # [L,U]  = tm_1x6_eigvals(dt_1x6);
    # ad     = (real(L(:,1)), 1);
    # rd     = (mean(real(L(:,2:3)),2), 1);
    # u      = (U,3);


    # copied from tmp_ct_to_dps.m
    # DTD paramters derived from first and second cumulant term
    V_MD = np.clip(tp(c4, Ebulk), 0, np.inf)
    V_iso = np.clip(tp(c4, E4iso), 0, np.inf)

    V_MD1 = V_MD + V_MD2
    V_iso1 = V_iso + V_iso2

    V_shear = np.clip(tp(c4, Eshear), 0, np.inf)
    V_shear1 = V_shear + V_shear2 # eq 22 Westin2016

    # ********* Normalized variance measures ************
    C_MD = V_MD / max(V_MD1, reg)
    C_mu = 1.5 * V_shear1 / max(V_iso1, reg)
    C_M = 1.5 * V_shear2 / max(V_iso2, reg)
    C_c = C_M / max(C_mu, reg)

    # ********* Kurtosis measures ************
    # Naming these according to the dtd_gamma nomenclature
    MKi = 3 * V_MD / max(V_MD2, reg)
    MKa = (6/5.) * V_shear1 / max(V_MD2, reg) # K_micro in Westin16
    MKt = MKi + MKa
    MKad = (6/5.) * V_shear / max(V_MD2, reg) # anisotropy and dispersion
    MK = MKad + MKi # conventional kurtosis
    MKd = MKa - MKad # conventional kurtosis
    uFA = np.sqrt(C_mu)

    S_I = np.sqrt(V_MD * (V_MD > 0))
    S_A = np.sqrt(V_shear1 * (V_shear1 > 0))

    return C_MD, C_mu, C_M, C_c, MD, FA, uFA
