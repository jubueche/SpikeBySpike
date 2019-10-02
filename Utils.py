import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DYNAPS/"))
from helper import signal_to_spike_refractory

class Utils:

    def __init__(self, Nneuron, Nx, lam, dt, epsr, epsf, alpha, beta, mu, gamma, Thresh, Nit, Ntime, A, sigma, delta_modulator_threshold):
        self.Nneuron = Nneuron
        self.Nx = Nx
        self.lam = lam
        self.dt = dt
        self.epsr = epsr
        self.epsf = epsf
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        self.Thresh = Thresh
        self.Nit = Nit
        self.Ntime = Ntime
        self.A = A
        self.sigma = sigma
        self.T = int(np.floor(np.log(self.Nit*self.Ntime)/np.log(2)))
        self.delta_modulator_threshold = delta_modulator_threshold


    @classmethod
    def from_json(self, dict):
        return Utils(Nneuron=dict["Nneuron"], Nx=dict["Nx"], lam=dict["lam"], dt=dict["dt"],
                epsr=dict["epsr"], epsf=dict["epsf"], alpha=dict["alpha"], beta=dict["beta"],
                mu=dict["mu"], gamma=dict["gamma"], Thresh=dict["Thresh"], Nit=dict["Nit"],
                Ntime=dict["Ntime"], A=dict["A"], sigma=dict["sigma"], delta_modulator_threshold=dict["delta_modulator_threshold"])

    def continous_to_spikes(self, x):
        ups = []; downs = []
        for i in range(x.shape[0]):
                tmp = signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], self.delta_modulator_threshold, self.delta_modulator_threshold, 0.0001)
                ups.append(np.asarray(tmp[0]))
                downs.append(np.asarray(tmp[1]))

        ups = np.asarray(ups)
        downs = np.asarray(downs)
        return (ups, downs)



def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def get_input(A, Nx, TimeL, w):
    InputL = A*(np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)).T
    for d in range(Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')
    return InputL

def ups_downs_to_O(ups, downs):
    Input_spikes = np.zeros((2*len(ups), utils.Ntime))
    for i in range(len(ups)):
        Input_spikes[i,np.asarray(ups[i], dtype=int)] = 1
        Input_spikes[i+len(ups), np.asarray(downs[i], dtype=int)] = 1

    return Input_spikes