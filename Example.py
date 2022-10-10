import numpy as np
from Definitions import DT_orientation, DT_evecs, DT_evals, FA_gen, MD_gen, Diffusion_Tensors_manual
from Definitions import b_ten_orien, b_tensors

' set up the set of diffusion tensors (diffusion tensor distribution) '
N = 100
k = 50
mu = [1., 0., 0.]
threshold = 0.8
D_shape = 'lin'

dt_orien = DT_orientation(N, k, mu, threshold)
dt_evecs = DT_evecs(N, dt_orien)

FA_dt = FA_gen(N, 0.8, 0.01)
MD_dt = MD_gen(N, 1.7*10*-3, 0.01)

dt_evals = DT_evals(D_shape, FA_dt, MD_dt)

# final Diffusion Tensor Distribution
DT_dis = Diffusion_Tensors_manual(dt_evecs, dt_evals)


' generate a set b-tensors '
N_bt = 100
b_vals=np . arange ( 0, 1000, N_bt) # s e t up b−v alu es in s /mm∗∗2
N_b_orien = 30 # number of b−t e sn o r o r i e n t a t i o n s
bt_orien = b_ten_orien(N_b_orien ) # b−t en s o r o r i e n t a t i o n s

def build_btens (N, bvals , orient , shape):
    bt_list = []
    for i in range(len(b_vals)):
        bt_list.append(b_tensors(N, bvals[i],orient,B_shape=shape))
        return np.concatenate(np.asarray(bt_list),axis =0)

# generate a set of linear and planar b−tensors
bt_lin = build_btens(N_bt, b_vals , bt_orien , 0.99)
bt_pla = build_btens (N_bt, b_vals , bt_orien ,-0.49)
bt = np.concatenate((bt_lin, bt_pla), axis=0)

