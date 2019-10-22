import numpy as np  
import math
import warnings

def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = np.random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)

def stochastic_round(C_real, F, min=-0.339, max=0.412):
    

    dynapse_maximal_synapse_o = 10
    
    np.fill_diagonal(C_real, 0)
    
    if(np.min(C_real) < min or np.max(C_real) > max):
        w = ("Recurrent matrix exceeds minimum or maximum. Max: %.3f, Min: %.3f" % (np.max(C_real),np.min(C_real)))
        warnings.warn(w, RuntimeWarning)

    # All elements that are bigger than max will be set to max, same for min
    C_real[C_real > max] = max
    C_real[C_real < min] = min

    # Scale the new weights with respect to the range
    C_new_discrete = np.zeros(C_real.shape)
    
    #! Bin here
    hist, bin_edges = np.histogram(C_real.reshape((-1,1)), bins = 2*dynapse_maximal_synapse_o, range=(min,max))
    C_new_discrete = np.digitize(C_real.ravel(), bins = bin_edges, right = True).reshape(C_new_discrete.shape) - dynapse_maximal_synapse_o
    
    assert (C_new_discrete <= dynapse_maximal_synapse_o).all() and (C_new_discrete >= -dynapse_maximal_synapse_o).all(), "Error, have value > or < than max/min in Omega"
    
    number_available_per_neuron = 62 - np.sum(np.abs(F), axis=1)

    for idx in range(C_new_discrete.shape[0]):
        num_available = number_available_per_neuron[idx]
        num_used = np.sum(np.abs(C_new_discrete[idx,:]))
        while(num_used > num_available):
            ind_non_zero = np.nonzero(C_new_discrete[idx,:])[0]
            rand_ind = np.random.choice(ind_non_zero, 1)[0]
            if(C_new_discrete[idx,rand_ind] > 0):
                C_new_discrete[idx,rand_ind] -= 1
            else:
                C_new_discrete[idx,rand_ind] += 1
            num_used -= 1

    assert ((number_available_per_neuron - np.sum(np.abs(C_new_discrete), axis=1)) >= 0).all(), "More synapses used than available"

    #! Reduce weights here

    # Stochastic round
    return C_new_discrete


# Obtained from running simulation for 140 iterations

dynapse_maximal_synapse_ff = 5
Nx = 2
Nneuron = 20
gamma = 1.0
np.random.seed(42)

F = 0.5*np.random.randn(Nx, Nneuron)
# F matrix is normalized
F = gamma*np.divide(F, np.sqrt(np.matmul(np.ones((Nx,1)), np.sum(F**2, axis=0).reshape((1,Nneuron)))))
M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
FtM = np.matmul(F.T, M)
for i in range(FtM.shape[1]): # for all columns
    divisor = (max(FtM[:,i]) - min(FtM[:,i]))
    if (divisor != 0):
        FtM[:,i] = FtM[:,i] / divisor * 2*dynapse_maximal_synapse_ff
FtM = np.asarray(FtM, dtype=int)

C_real = np.random.rand(Nneuron,Nneuron)-0.5*np.eye(Nneuron)

new_discrete_C = stochastic_round(C_real, FtM, min=-0.339, max=0.412)
