import numpy as np  
from AudioHelper import AudioHelper
import matplotlib.pyplot as plt
from scipy.signal import resample

audio = AudioHelper()
t,sample = audio.get_random_sample()

factor = int(t / 1000)
subsamples_n = int(len(sample) / factor)
print("Subsampling to %d" % subsamples_n)

"""plt.plot(sample)
plt.show()"""

"""S = np.fft.fft(sample)
freq = np.fft.fftfreq(sample.size, d=1/500)

plt.plot(freq, S)
plt.show()"""

resampled = resample(sample, subsamples_n)

plt.plot(resampled)
plt.show()

