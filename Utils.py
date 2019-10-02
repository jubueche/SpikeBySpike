import numpy as np

class Utils:

    def __init__(self, Nneuron, Nx, lam, dt, epsr, epsf, alpha, beta, mu, gamma, Thresh, Nit, Ntime, A, sigma):
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


    @classmethod
    def from_json(self, dict):
        return Utils(Nneuron=dict["Nneuron"], Nx=dict["Nx"], lam=dict["lam"], dt=dict["dt"],
                epsr=dict["epsr"], epsf=dict["epsf"], alpha=dict["alpha"], beta=dict["beta"],
                mu=dict["mu"], gamma=dict["gamma"], Thresh=dict["Thresh"], Nit=dict["Nit"],
                Ntime=dict["Ntime"], A=dict["A"], sigma=dict["sigma"])



def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def get_input(A, Nx, TimeL, w):
    InputL = A*(np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)).T
    for d in range(Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')
    return InputL