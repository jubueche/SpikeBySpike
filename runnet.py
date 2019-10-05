import numpy as np  
from Utils import my_max
import json
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "DYNAPS/"))
from helper import signal_to_spike_refractory

def runnet(dt, lam, F, Input, C, Nneuron, Ntime, Thresh):

    with open(os.path.join(os.getcwd(), "parameters.param"), 'r') as f:
            parameters = json.load(f)

    r0 = np.zeros((Nneuron, Ntime))
    O = np.zeros((Nneuron, Ntime))
    V = np.zeros((Nneuron, Ntime))
    
    for t in range(1, Ntime):
        V[:,t] = ((1-lam*dt)*V[:,t-1].reshape((-1,1)) + dt*np.matmul(F.T, Input[:,t-1].reshape((-1,1))) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron,1)).ravel()
        (m,k) = my_max(V[:,t].reshape((-1,1)) - Thresh-0.01*np.random.randn(Nneuron, 1))

        if(m>=0):
            O[k,t] = 1
        
        r0[:,t] = ((1-lam*dt)*r0[:,t-1].reshape((-1,1))+1*O[:,t].reshape((-1,1))).ravel()

    return (r0, O, V)


def runnet_spike_input(dt, lam, conn_x_high, conn_x_down, OT_up, OT_down, C, Nneuron, Ntime, Thresh):
    # C should be Identity with diag(I) = -0.5
    r0 = np.zeros((Nneuron, Ntime))
    O = np.zeros((Nneuron, Ntime))
    V = np.zeros((Nneuron, Ntime))
    
    for t in range(1, Ntime):
        spiking_input_v_cont = conn_x_high[0]*OT_up[0,t] + conn_x_high[1]*OT_up[1,t]-conn_x_down[0]*OT_down[0,t]-conn_x_down[1]*OT_down[1,t]

        V[:,t] = ((1-lam*dt)*V[:,t-1].reshape((-1,1)) + spiking_input_v_cont.reshape((-1,1)) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron, 1)).ravel()
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

