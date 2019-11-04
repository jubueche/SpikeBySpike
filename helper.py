import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
import matplotlib.pyplot as plt
from scipy import interpolate
        
def get_input(A, Nx, lam, dt, duration=1000):
    xT = np.zeros((Nx, duration))
    w = np.load("Resources/w.dat", allow_pickle=True)
    InputT = A*(np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), duration)).T

    for d in range(Nx):
        InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

    return InputT


def signal_to_spike_refractory(interpfact, time, amplitude, thr_up, thr_dn, refractory_period):
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
