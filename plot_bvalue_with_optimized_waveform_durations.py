import matplotlib.pyplot as plt
import numpy as np

pla_dur = np.arange(12, 22, 0.5)
pla_bval = np.array([9330.09, 11177.98, 12587.66, 14091.64, 14905.09, 16595.43, 18409.31, 20350.1, 22422.84, 24631.4,
                     26980.23, 29473.69, 32115.89, 34911.38, 37864.2, 40978.88, 43154.84, 47710.49, 51336.33, 55140.84])

lin_dur = np.arange(12, 14, 0.5)
lin_bval = np.array([33750.43, 37220.64, 41366.07, 45394.98])

plt.figure(figsize=(10, 6))
plt.plot(pla_dur, pla_bval,'o', label='planar b-tensor')
plt.plot(lin_dur,  lin_bval,'s',label='linear b-tensor')
plt.ylabel('b-value [s/mm²]', fontsize=15)
plt.xlabel('gradient waveform duration [ms]',fontsize=15)
plt.title('Diffusion weighting gradient waveform duration and corresponding b-value\n for a linear and planar encoding scheme',fontsize=15)
plt.xticks(np.arange(12., 22., 1.0), fontsize=12)
plt.yticks(np.arange(10000, 56000, 5000), fontsize=12)
plt.legend(loc='lower right', fontsize=15)
plt.grid()
plt.show()