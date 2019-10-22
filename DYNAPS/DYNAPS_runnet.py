import numpy as np
import matplotlib.pyplot as plt

def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def runnet(sbs, utils, F, C_dis, C_real, duration, x, x_recon_lam = 0.001, x_recon_R = 1.0, delta_F=0.1):
    # Set the recurrent weights
    O_DYNAPS_tmp = np.zeros((utils.Nneuron, duration))
    R = np.zeros((utils.Nneuron, duration))
    V_recons = np.zeros((utils.Nneuron, duration))

    # Set the matrix C by using a delta of 0
    sbs.set_recurrent_weight_directly(C_dis)
    sbs.set_recurrent_connection()

    O_DYNAPS_tmp = sbs.execute()

    for t in range(1, duration):
        neurons_that_spiked = np.nonzero(O_DYNAPS_tmp[:,t])[0]
        r_tmp = np.copy(R[:,t-1])
        r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked))
        R[:,t] = (1-utils.lam*utils.dt)*r_tmp

    for t in range(1, duration):
        V_recons[:,t] = 0.1*V_recons[:,t-1] + delta_F*np.matmul(F.T, x[:,t]) + np.matmul(C_real, R[:,t-1])

    if(sbs.debug):
        plt.plot(V_recons[5,:])
        plt.show()
        plt.plot(R[5,:])
        plt.show()

    """for t in range(1, utils.Ntime):
        current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
        new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, x[:,t]) + np.matmul(C_real, R[:,t-1])

        (m, k) = my_max(new_V_recon.reshape((-1,1)) - current_thresh) # Returns maximum and argmax

        if(m[0] >= 0 and O_DYNAPS_tmp[k,t]): # Check if we actually get a spike from DYNAPS
            # Update rate vector
            r_tmp = R[:,t-1]
            r_tmp[k] += 1
            R[:,t] = (1-utils.lam*utils.dt)*r_tmp
        V_recons[:,t] = new_V_recon"""

    
    return (R, O_DYNAPS_tmp, V_recons)
