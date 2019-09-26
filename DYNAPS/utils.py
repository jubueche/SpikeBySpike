import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
from matplotlib.pyplot import *
from scipy import interpolate


class Utils:

    def __init__(self, duration, delta_t, dtt, eta, n_in, sigma_x):
        
        self.duration = duration
        self.delta_t = delta_t # ms
        self.dtt = dtt
        self.eta  = eta
        self.time_steps = int(self.duration/self.delta_t)
        self.n_in = n_in
        self.sigma_x = sigma_x
        
    @classmethod
    def from_default(self):
        return Utils(duration=8000, delta_t=1, dtt=10**-3, eta=1000, n_in=2, sigma_x=1)


    def set_duration(self, duration):
        self.duration = duration
        self.time_steps = int(self.duration/self.delta_t)

    def get_matlab_like_input(self):
        T = self.duration
        dim = self.n_in
        seed = np.random.randn(dim, T)*self.sigma_x
        L = round(6*self.eta)
        cNS = np.hstack([np.zeros((dim,1)), seed])

        ker = np.exp( -((np.linspace(1,L,L) - L/2))**2/self.eta**2)
        ker = ker/sum(ker)

        x = np.zeros((dim, max([cNS.shape[1]+len(ker)-1,len(ker),cNS.shape[1] ])))

        for i in range(0,dim):
            x[i,:] = np.convolve(cNS[i,:], ker)*np.sqrt(self.eta/0.4)

        x = x[:, 0:T]
        return x


    def signal_to_spike_refractory(self, interpfact, time, amplitude, thr_up, thr_dn,refractory_period):
        #interpfact: desired frequency of the upsampled data
        actual_dc = 0 
        spike_up = []
        spike_dn = []

        f = interpolate.interp1d(time, amplitude)                
        rangeint = np.round((np.max(time) - np.min(time))*interpfact)
        xnew = np.linspace(np.min(time), np.max(time), num=int(rangeint), endpoint=True)                
        data = np.reshape([xnew, f(xnew)], (2, len(xnew))).T
        
        i = 0
        while i < (len(data)):
            if( (actual_dc + thr_up) < data[i,1]):
                spike_up.append(data[i,0] )  #spike up
                actual_dc = data[i,1]        # update current dc value
                i += int(refractory_period * interpfact)
            elif( (actual_dc - thr_dn) > data[i,1]):
                spike_dn.append(data[i,0] )  #spike dn
                actual_dc = data[i,1]       # update curre
                i += int(refractory_period * interpfact)
            else:
                i += 1

        return spike_up, spike_dn


    def save_F(self, direc, X, **kwargs):
        figsize=(np.asarray(X.shape))[::-1]
        rcParams.update({'figure.figsize':figsize})
        fig = figure(figsize=figsize)
        axes([0,0,1,1]) # Make the plot occupy the whole canvas
        axis('off')
        fig.set_size_inches(figsize)
        imshow(X,origin='lower', **kwargs)
        savefig(direc, facecolor='black', edgecolor='black', dpi=100)
        close(fig)

    # Spike times in millis
    def spikes_to_isi(self, spike_times, neurons_id, use_microseconds=True):
        if(use_microseconds):
            print("Using microseconds. (1e-6s)")
        else:
            print("Using 10s of microseconds (1e-5s)")

        Signal_isi = []
        for i in range(len(spike_times)):
            if i == 0 :
                Signal_isi.append(spike_times[0])
            else:
                Signal_isi.append(spike_times[i] - spike_times[i-1])
        Signal_isi = np.asarray(Signal_isi)
        if(use_microseconds):
            Signal_isi = Signal_isi * 1e3
        else:
            Signal_isi = Signal_isi * 1e2

        # Avoid using neuron zero (because all neurons are connected to it)
        if(0 in neurons_id):
            neurons_id = neurons_id + 1 
        return (Signal_isi, neurons_id)