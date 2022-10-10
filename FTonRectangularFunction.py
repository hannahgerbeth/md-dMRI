#https://www.youtube.com/watch?v=98VixA3MjAc
import numpy as np
import matplotlib.pyplot as plt

L = 10
N = 1024
dx = L/(N-1)
x = np.linspace(0,L,N) #?

f = np.zeros(len(x))
f[256:768] = 1

A0 = np.sum(np.dot(f,np.ones(len(x))))*dx*2/L

fFS_10 = A0/2
for i in range(10):
    Ak = np.sum(np.dot(f, np.cos(2 * i * np.pi * x / L))) * dx * 2 / L
    Bk = np.sum(np.dot(f, np.sin(2 * i * np.pi * x / L))) * dx * 2 / L
    fFS_10 = fFS_10 + Ak * np.cos(2 * i * np.pi * x / L) + Bk * np.sin(2 * i * np.pi * x / L)

fFS_100 = A0/2
for k in range(100):
    Ak = np.sum(np.dot(f,np.cos(2*k*np.pi*x/L)))*dx*2/L
    Bk = np.sum(np.dot(f,np.sin(2*k*np.pi*x/L)))*dx*2/L
    fFS_100 = fFS_100 + Ak*np.cos(2*k*np.pi*x/L) + Bk*np.sin(2*k*np.pi*x/L)

plt.plot(x,f, 'k-', linewidth=4, label='f(x): discontinous function')
plt.plot(x,fFS_10 -1, '-', linewidth=3, label='F(x): Fourier Series with k=10')
plt.plot(x,fFS_100 -1, 'r-', linewidth=3, label='F(x): Fourier Series with k=100')
plt.legend()
#plt.gcf('Position', [1500, 200, 2500, 1500])
plt.show()



x_new = np.arange(0,2*np.pi,0.01)   # start,stop,step
labels = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$',
          r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$']

fig, axs = plt.subplots(5)
#fig, axs = plt.subplots(nrows=6,
#                         gridspec_kw={"height_ratios" : [1,1,1,1,.5,1], "hspace":0})

plt.rcParams.update({'font.size': 12})
fig.suptitle('Superposition of sine functions and the corresponding Fourier transform')
axs[0].plot(x_new, np.sin(x_new), label='sin(x): f=1 Hz, A=1')
axs[0].legend(loc='upper right')
#axs[0].tick_params(axis="x", bottom=False, labelbottom=False)
axs[0].set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
axs[0].set_xticklabels(labels)

axs[1].plot(x_new, np.sin(3*x_new), label='sin(3x): f=3 Hz, A=1')
axs[1].legend(loc='upper right')
#axs[1].tick_params(axis="x", bottom=False, labelbottom=False)
axs[1].set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
axs[1].set_xticklabels(labels)

axs[2].plot(x_new, 2*np.sin(8*x_new), label='2sin(8x): f=8 Hz, A=2')
axs[2].legend(loc='upper right')
#axs[2].tick_params(axis="x", bottom=False, labelbottom=False)
axs[2].set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
axs[2].set_xticklabels(labels)

axs[3].plot(x_new, np.sin(x_new) + np.sin(3*x_new) + 2*np.sin(8*x_new), label='sin(x) + sin(3x) + 2sin(8x)')
axs[3].legend(loc='upper right')
axs[3].tick_params(axis="x", bottom=True, labelbottom=True)
axs[3].set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
axs[3].set_xticklabels(labels)
axs[3].set_xlabel('x')

#plt.show()

from scipy.fft import fft, fftfreq


y = np.sin(x_new) + np.sin(3*x_new) + 2*np.sin(8*x_new)

N = 1000
dx = 1/N
xf = fftfreq(N, dx)

yf = fft(y)
xf = fftfreq(N, dx)[:N//2]

axs[4].plot(xf, 2.0/N * np.abs(yf[0:N//2]), label='FT[sin(x) + sin(3x) + 2sin(8x)]')
axs[4].legend(loc='upper right')
axs[4].set_xlabel('frequency [Hz]')
plt.xlim((0,100))
plt.grid()
plt.show()

from scipy.fft import dct, idct
import matplotlib.pyplot as plt
N = 100
t = np.linspace(0,20,N, endpoint=False)
x = np.exp(-t/3)*np.cos(2*t)

plt.plot(t, x, '-b', label='continuous function')

N = 50
t = np.linspace(0,20,N, endpoint=False)
x = np.exp(-t/3)*np.cos(2*t)
y = dct(x, norm='ortho')
window = np.zeros(N)
window[:20] = 1
yr = idct(y*window, norm='ortho')

plt.plot(t, yr, 'ro', label='sampled discrete data points')
plt.legend()
plt.xlabel('time [s]')

plt.rcParams.update({'font.size': 12})
plt.grid()
plt.show()
