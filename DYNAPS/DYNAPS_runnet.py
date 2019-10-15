import numpy as np 

def runnet(sbs, utils, F, C, duration, x, x_recon_lam = 0.001, x_recon_R = 1.0, delta_F=0.1):
    # Set the recurrent weights
    O_DYNAPS = np.zeros((utils.Nneuron, duration))
    R = np.zeros((utils.Nneuron, duration))
    V_recons = np.zeros((utils.Nneuron, duration))

    sbs.set_reccurent_connection(np.copy(C), discretized = False)
    O_DYNAPS = sbs.execute()

    for t in range(1, duration):
        neurons_that_spiked = np.nonzero(O_DYNAPS[:,t])[0]
        r_tmp = R[:,t-1]
        r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked)) 
        R[:,t] = (1-utils.lam*utils.dt)*r_tmp

    for t in range(1, duration):
        V_recons[:,t] = 0.1*V_recons[:,t-1] + np.matmul(F.T, x[:,t]) + np.matmul(C, R[:,t-1])
    
    return (R, O_DYNAPS, V_recons)