import numpy as np
import matplotlib.pyplot as plt

def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def runnet(sbs, utils, F, C_dis, duration, x, x_recon_lam = 0.001, x_recon_R = 1.0, delta_F=0.1):
    # Set the recurrent weights
    O_DYNAPS_tmp = np.zeros((utils.Nneuron, duration))
    R = np.zeros((utils.Nneuron, duration))
    V_recons = np.zeros((utils.Nneuron, duration))

    # Set the matrix C by using a delta of 0
    sbs.set_recurrent_weight_directly(C_dis)
    sbs.set_recurrent_connection()

    O_DYNAPS_tmp = sbs.execute()

    if(sbs.debug):
        coordinates = np.nonzero(O_DYNAPS_tmp)
        plt.figure(figsize=(18, 6))
        plt.scatter(coordinates[1], coordinates[0]+1, marker='o', s=0.5, c='k')
        plt.ylim((0,utils.Nneuron+1))
        plt.yticks(ticks=np.linspace(0,utils.Nneuron,int(utils.Nneuron/2)+1))
        plt.show()

    for t in range(1, duration):
        neurons_that_spiked = np.nonzero(O_DYNAPS_tmp[:,t])[0]
        r_tmp = np.copy(R[:,t-1])
        r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked))
        R[:,t] = (1-utils.lam*utils.dt)*r_tmp

    for t in range(1, duration):
        V_recons[:,t] = 0.1*V_recons[:,t-1] + np.matmul(F.T, x[:,t]) + np.matmul(C_dis, R[:,t-1])
    
    return (R, O_DYNAPS_tmp, V_recons)
