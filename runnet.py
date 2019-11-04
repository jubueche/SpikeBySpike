import numpy as np  
from Utils import my_max
from helper import signal_to_spike_refractory

def runnet(utils,dt, lam, F, Input, C, Nneuron, duration, Thresh, update_all = False, use_spiking = False):
    r0 = np.zeros((Nneuron, duration))
    O = np.zeros((Nneuron, duration))
    V = np.zeros((Nneuron, duration))

    I = np.zeros((2*utils.Nx, 1))
    M = np.asarray([[1,-1,0,0],[0,0,1,-1]])
    x_recon_lam = 0.001
    x_recon_R = 40.3

    if(use_spiking):
        (OT_down, OT_up) = get_spiking_input(30, Input, utils.Nx, duration)

    for t in range(1, duration):
        if(use_spiking):
            ot_in = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
            I = (1-x_recon_lam)*I + x_recon_R*ot_in
            FTMI = np.matmul(np.matmul(F.T, M), I)
            V[:,t] = ((1-utils.lam*utils.dt)*V[:,t-1].reshape((-1,1)) + utils.dt*FTMI.reshape((-1,1)) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron,1)).ravel()
        else:    
            V[:,t] = ((1-lam*dt)*V[:,t-1].reshape((-1,1)) + dt*np.matmul(F.T, Input[:,t-1].reshape((-1,1))) + np.matmul(C, O[:,t-1].reshape((-1,1))) + 0.001*np.random.randn(Nneuron,1)).ravel()
        (m,k) = my_max(V[:,t].reshape((-1,1)) - Thresh-0.01*np.random.randn(Nneuron, 1))

        neurons_that_spiked = (V[:,t].reshape((-1,1)) - Thresh-0.01*np.random.randn(Nneuron, 1)) >= 0
        if(update_all):
            O[neurons_that_spiked.ravel(),t] = 1.0
        elif (m >= 0):
            O[k,t] = 1
        
        r0[:,t] = ((1-lam*dt)*(r0[:,t-1].reshape((-1,1))+1*O[:,t].reshape((-1,1)))).ravel()

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