import numpy as np
import nibabel as nib
import scipy

def load_data(filename):
    data_img = nib.load(f'{filename}')
    data = data_img.get_fdata()
    affine = data_img.affine
    return data, affine

def readfile_btens(filename):
    a = np.loadtxt(filename)
    return np.array([np.reshape(a[i], (3, 3)) for i in range(len(a))])


# reorient the btensors so that they match with data
def reorient_btensors(input_btensors, a, b):
    # input the btensors and the two axis that should be swapped/changed
    # e.g. (0, 1, 2) -> (0, 2, 1), then a=1 and b=2
    btensors_reoriented = np.zeros(input_btensors.shape)
    for i in range(input_btensors.shape[0]):
        temp = input_btensors[i, :, :]
        temp[[a,b], :] = temp[[b,a], :] #change rows
        temp[:, [a,b]] = temp[:, [b,a]] #change columns
        btensors_reoriented[i] = temp
    return btensors_reoriented

def reshape_pad(data, fname, axis=(0, 2, 1), pad = 5):
    data_reshape = np.flip(np.flip(np.transpose(data, axis), axis=2), axis=0)
    data_reshape_pad = np.pad(data_reshape, (int(pad)),  'constant')

    if len(data_reshape.shape) > 3:
        data_reshape_pad = data_reshape_pad[..., int(pad):-int(pad)]

    #nib.Nifti1Image(data_reshape_pad, affine).to_filename(f'{fname}.nii')
    return data_reshape_pad


def mean_bvals(bvals):
    bvals_list = np.round(bvals, decimals=6).tolist()
    temp_bvals_list = []
    for i in bvals_list:
        if i not in temp_bvals_list:
            temp_bvals_list.append(i)
    #return np.unique(bvals_list)
    return np.array(temp_bvals_list)

def mean_signal(data, bvals):
    bvals_mean = mean_bvals(bvals)
    signal_mean = np.zeros(bvals_mean.shape[0])
    for i in range(bvals_mean.shape[0]):
        signal_mean[i] = np.mean(data[np.where(np.round(bvals, decimals=6) == bvals_mean[i])])
    return signal_mean

def mean_Sb(data, bvals):
    bvals_mean = mean_bvals(bvals)
    signal_mean = mean_signal(data, bvals)
    return bvals_mean, signal_mean

def monoexp(x, b, c):
    return (1 - c) * np.exp(- b * x) + c

def monoexp_fit(data, bvals):

    # fit: f(x) = a * exp(-bx) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1
    # f(x) = (1-c) * exp(-bx) + c
    # -> S(b) = (1-df) * e^(-b*MD) + df
    # df = dot-fraction

    fit_bounds = ([0, 0], [np.inf, 1])

    # initialization
    # ln(S)/(-b) = ADC -> ADC depends on b and is a vector -> take mean of ADC to initialize MD
    # Suppress/hide the warning 'invalid value encountered in true_divide'
    np.seterr(invalid='ignore')

    tmp1 = np.log(data)/(- bvals)
    #MD_init = np.mean(tmp1[np.where(np.isnan(tmp1)==False)]) # if there are nan-value do not take them into account
    MD_init = np.mean(tmp1[1:])

    # take minimum of signal to initialize dot-fraction c
    if np.min(data)<=0:
        dot_init = 1e-5
    else:
        dot_init = np.min(data)

    init = ([MD_init, dot_init])

    def jac_monoexp(bs, *coef):
        c,b = coef

        d_c = 1-np.exp(-bs*b)
        d_b = (c-1)*bs*np.exp(-bs*b)

        return np.array([d_c, d_b]).T

    try:
        #tmp, pcov = scipy.optimize.curve_fit(monoexp, bvals, data, p0=init, bounds=fit_bounds, maxfev=3000)
        tmp, pcov = scipy.optimize.curve_fit(monoexp, bvals, data, p0=init,jac=jac_monoexp, bounds=fit_bounds, maxfev=3000)
    except RuntimeError:
        tmp = [0, 0]
        pcov = [0]
        print("Error - curve_fit failed: monoexp_fit")

    return tmp, pcov, init

def bi_exp(b, MD1, MD2, frac1, dotfrac):
    # E0 = (1-dotfrac)*[frac1*exp(-bs*MD1) + (1-frac1)*exp(-bs*MD2)] + dotfrac
    # 0 <= frac1, dotfrac <= 1
    # dotfrac + (1-dotfrac)*frac1 + (1-dotfrac)(1-frac1) = 1
    return (1 - dotfrac) * (frac1 * np.exp(-b * MD1) + (1 - frac1) * np.exp(-b * MD2)) + dotfrac

def biexp_fit(data, bvals, para_mono):

    # fit: f(x) = a1 * exp(-b1x) + a2 * exp(-b2x) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1
    # f(x) = (1-c) * [a exp(-b1x) + (1-a) exp(-b2x)] + c
    # -> S(b) = frac_fast * e^(-bD_fast) + frac_slow e^(-bD_slow) + dotfrac

    fit_bounds = ([0, 0, 0, 0], [np.inf, np.inf, 1, 1])

    # mono-exp fit to get initial guesses -> gives df, MD1
    # set initial guesses: MD1 = MD1_mono, MD2=MD1_mono/2, df = df_mono,
    # f(x) = (1-c) * exp(-b1x) + (1-c-a) exp(-b2x) + c
    # -> S(b) = frac_fast * e^(-bD_fast) + frac_slow e^(-bD_slow) + dotfrac
    # initial guesses: MD1 = tmp_init[0], MD2=tmp_init[0]/2, frac1=tmp_init[1], dotfrac=tmp_init[1]/2
    init = ([para_mono[0], para_mono[0]/2, para_mono[1]/2, para_mono[1]])

    def jac_biexp(bs, *coef):
        MD1, MD2, frac1, dotfrac = coef

        d_MD1 = (dotfrac - 1)*frac1*bs*np.exp(-bs*MD1)
        d_MD2 = -(dotfrac -1)*(frac1 -1)*bs*np.exp(-MD2*bs)
        d_frac1 = (1-dotfrac)*(np.exp(-MD1*bs) - np.exp(-MD2*bs))
        d_dotfrac = - frac1*np.exp(-MD1*bs) + (frac1-1)*np.exp(-MD2*bs) +1

        return np.array([d_MD1, d_MD2, d_frac1, d_dotfrac]).T

    try:
        #tmp, pcov = scipy.optimize.curve_fit(bi_exp, bvals, data, p0=init, bounds=fit_bounds, maxfev=3000)
        tmp, pcov = scipy.optimize.curve_fit(bi_exp, bvals, data, p0=init, jac=jac_biexp, bounds=fit_bounds,maxfev=3000)
    except RuntimeError:
        print("Error - curve_fit failed: biexp_fit")
        tmp = [0, 0, 0, 0]
        pcov = [0]

    return tmp, pcov, init

def cumulant_exp(x, MD, MK, c):
    return (1 - c) * np.exp(- MD * x + 0.5 * MK * x ** 2) + c
    # return np.exp(- a*x + 0.5 * b**2 * x) + c

def cumexp_fit(data, bvals, para_mono):

    # fit: f(x) = (1-c) * exp(-b MD + 0.5 b^2 MK) + c
    # constraint all fractions to [0,1] with Sum(fractions)=1

    fit_bounds = ([0, -np.inf, 0], [np.inf, np.inf, 1])

    # S = exp(- b*MD + 0.5*b**2 * MK)
    # ln(S) = - b*MD + 0.5*b**2 * MK
    # ln(S)/b = b*0.5*MK - MD -> a = 0.5 MK, b = -MD
    y = np.log(data)/bvals
    x = bvals
    # remove possible (first digit) nan-values by just fitting tmp1[:1]
    a,b = np.polyfit(x[1:],y[1:],1)
    #
    MD_init = - b
    MK_init = a/0.5
    # take dotfrac_init from mono-exp fit
    df_init = para_mono[1]

    init = ([MD_init, MK_init, df_init])

    def jac_cumulant(bs, *coef):
        MD, MK, c = coef

        d_MD = (c-1)*bs*np.exp(-bs*(MD-0.5*MK*bs))
        d_MK = -0.5 * (c-1) * bs**2 *np.exp(-bs*(MD-0.5*MK*bs))
        d_c = 1-np.exp(bs*(0.5*MK*bs - MD))

        return np.array([d_MD, d_MK, d_c]).T

    #plt.plot(data)
    #plt.plot(np.exp(- bvals * MD_init + 0.5 * bvals ** 2 * MK_init))
    #plt.show()

    try:
        #tmp, pcov = scipy.optimize.curve_fit(cumulant_exp, bvals, data, p0=init, bounds=fit_bounds,maxfev=5000)
        tmp, pcov = scipy.optimize.curve_fit(cumulant_exp, bvals, data, p0=init, jac=jac_cumulant, bounds=fit_bounds, maxfev=5000)
    except RuntimeError:
        print("Error - curve_fit failed: cumexp_fit")
        tmp = [0, 0, 0]
        pcov = [0]

    return tmp, pcov, init


