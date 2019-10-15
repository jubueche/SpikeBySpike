import numpy as np  
from Utils import my_max
import json
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "DYNAPS/"))
from helper import signal_to_spike_refractory


def runnet_recon_x(dt, lam, F, OT_up, OT_down, C, Nneuron, Ntime, Thresh, x, x_recon_lam = 0.001, x_recon_R = 1.0, delta_F = 0.1):

    try:
        with open(os.path.join(os.getcwd(), "parameters.param"), 'r') as f:
                parameters = json.load(f)
    except:
        print("File not found. Looking in upper directory.")
        with open(os.path.join(os.getcwd(), "../parameters.param"), 'r') as f:
            parameters = json.load(f)

    Nx = F.shape[0]
    I = np.zeros((2*Nx, 1))
    M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    FTMI = np.zeros((Nneuron, 1))
    r0 = np.zeros((Nneuron, Ntime))
    O = np.zeros((Nneuron, Ntime))
    V = np.zeros((Nneuron, Ntime))
    V_recon = np.zeros((Nneuron, Ntime))
    x_recon = np.zeros((Nx, 1)) # (2,1)
    
    for t in range(1, Ntime):

        ot = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
        
        # V[:,t] = ((1-lam*dt)*V[:,t-1].reshape((-1,1)) + delta_F*FTMI.reshape((-1,1)) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron,1)).ravel()
        V[:,t] = 0.1*V[:,t-1] + np.matmul(F.T, x[:,t]) + np.matmul(C, r0[:,t-1]) + 0.001*np.random.randn(Nneuron,1).ravel()


        I = (1-x_recon_lam)*I + x_recon_R*ot
        FTMI = np.matmul(np.matmul(F.T, M), I)

        (m,k) = my_max(V[:,t].reshape((-1,1)) - Thresh-0.01*np.random.randn(Nneuron, 1))

        if(m>=0):
            O[k,t] = 1
        
        r0[:,t] = ((1-lam*dt)*r0[:,t-1].reshape((-1,1))+1*O[:,t].reshape((-1,1))).ravel()

    return (r0, O, V)


def get_spiking_input(threshold, Input, Nx, Ntime):
    # Compute spiking input
    ups = []; downs = []
    for i in range(Input.shape[0]):
            tmp = signal_to_spike_refractory(1, np.linspace(0,len(Input[i,:])-1,len(Input[i,:])), Input[i,:], threshold, threshold, 0.001)
            ups.append(np.asarray(tmp[0]))
            downs.append(np.asarray(tmp[1]))

    ups = np.asarray(ups)
    downs = np.asarray(downs)
    OT_up = np.zeros((Nx, Ntime))
    OT_down = np.zeros((Nx, Ntime))
    for i in range(Nx):
        OT_up[i,np.asarray(ups[i], dtype=int)] = 1
        OT_down[i,np.asarray(downs[i], dtype=int)] = 1

    return (OT_down, OT_up)

