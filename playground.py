import matplotlib.pyplot as plt
import numpy as np
from utils import Utils


utils = Utils()

T = utils.duration
dim = utils.n_in
seed = np.random.randn(dim, T)*utils.sigma_x
L = round(6*utils.eta)
cNS = np.hstack([np.zeros((dim,1)), seed])

ker = np.exp( -((np.linspace(1,L,L) - L/2))**2/utils.eta**2)
ker = ker/sum(ker)

x = np.zeros((dim, max([cNS.shape[1]+len(ker)-1,len(ker),cNS.shape[1] ])))

for i in range(0,dim):
    x[i,:] = np.convolve(cNS[i,:], ker)*np.sqrt(utils.eta/0.4)

x = x[:, 0:T]
