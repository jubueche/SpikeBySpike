import numpy as np  
from Utils import my_max
import json
import os


def runnet(dt, lam, F, Input, C, Nneuron, Ntime, Thresh):

    with open(os.path.join(os.getcwd(), "parameters.param"), 'r') as f:
            parameters = json.load(f)

    r0 = np.zeros((Nneuron, Ntime))
    O = np.zeros((Nneuron, Ntime))
    V = np.zeros((Nneuron, Ntime))
    I = np.zeros((Nneuron, Ntime))
    
    for t in range(1, Ntime):
        V[:,t] = ((1-lam*dt)*V[:,t-1].reshape((-1,1)) + dt*np.matmul(F.T, Input[:,t-1].reshape((-1,1))) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron,1)).ravel()
        (m,k) = my_max(V[:,t].reshape((-1,1)) - Thresh-0.01*np.random.randn(Nneuron, 1))

        if(m>=0):
            O[k,t] = 1
        
        r0[:,t] = ((1-lam*dt)*r0[:,t-1].reshape((-1,1))+1*O[:,t].reshape((-1,1))).ravel()

    return (r0, O, V)