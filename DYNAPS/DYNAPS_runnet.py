import numpy as np
import matplotlib.pyplot as plt

def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def runnet(sbs, utils, F, C_dis, C_real, duration, x, plot_dist = False, find_bias=False):
    # Set the recurrent weights
    O_DYNAPS = np.zeros((utils.Nneuron, duration))
    R = np.zeros((utils.Nneuron, duration))
    V_recons = np.zeros((utils.Nneuron, duration))

    if(not find_bias):
        sbs.set_recurrent_weight_directly(C_dis)
        sbs.set_recurrent_connection()

    O_DYNAPS = sbs.execute()
    spike_alignment_distribution = []

    if(not find_bias):
        for t in range(1, duration):
            V_recons[:,t] = 0.1*V_recons[:,t-1] + np.matmul(F.T, x[:,t]) + np.matmul(C_real, R[:,t-1])
            current_tresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
            (m,k) = my_max(V_recons[:,t] - current_tresh.ravel())
            has_spike = np.sum(O_DYNAPS[k,max(0,t-utils.alignment_delta_t):min(utils.Ntime-1,t+utils.alignment_delta_t)]) > 0

            spike_alignment_distribution.append(np.nonzero(O_DYNAPS[k,max(0,t-utils.alignment_delta_t):min(utils.Ntime-1,t+utils.alignment_delta_t)])[0])

            r_tmp = R[:,t-1]
            if(m >= 0 and has_spike):
                ot = np.zeros(utils.Nneuron)
                ot[k] = 1
                r_tmp[k] += 1
            R[:,t] = (1-utils.lam*utils.dt)*r_tmp

        if(plot_dist):
            plt.hist(spike_alignment_distribution, density=True)
            plt.show()
    
    return (R, O_DYNAPS, V_recons)
