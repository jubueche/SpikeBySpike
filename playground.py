import numpy as np  
import math

def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = np.random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)

def stochastic_round(C_real, delta_C_real, F_discrete, weight_range=(-1,1), debug = False):
    #! Change to utils.Nneuron, use utils.dynapse_maxi...
    """
    Given: C_real:       The recurrent connection matrix from the simulation
           delta_C_real: The delta of the recurrent connection matrix from the simulation
           F_discrete:   To check how many connections we are already using for each neuron
           weight_range: The range of the recurrent weights. These values are obtained from a previously run simulation
    Returns: Scaled and discretized, new weight matrix that has a maximum of 64 - max(FF) incoming neuron connections
    """
    C_new_real = C_real - delta_C_real
    dynapse_maximal_synapse_o = 5
    
    # Scale the new weights with respect to the range
    C_new_discrete = np.zeros(C_real.shape)
    for i in range(C_real.shape[0]):
        C_new_discrete[:,i] = C_new_real[:,i] / (weight_range[1] - weight_range[0]) * 2*dynapse_maximal_synapse_o    

    for i in range(C_real.shape[0]):
        for j in range(C_real.shape[0]):
            C_new_discrete[i,j] = prob_round(C_new_discrete[i,j])
    
    C_new_discrete = C_new_discrete.astype(np.int)

    # Number of neurons available should be equal for every neuron. Otherwise we would artifically increase the weight
    # of some neurons
    number_available = 64 - max(np.sum(np.abs(F_discrete), axis=1))
    if(debug):
        print("Number available per neuron %d" % number_available)

    # How many connections are we using now in total?
    
    total_num_used = np.sum(np.abs(C_new_discrete))
    if(debug):
        print("Number used %d / %d" % (total_num_used, C_real.shape[0]*number_available))
    difference = total_num_used - number_available*C_real.shape[0]
    if(debug):
        print("Difference is %d" % difference)

    
    # Need to reduce number of connections equally
    if(difference > 0):
        if(debug):
            print(C_new_discrete[:,0])
            number_to_reduce_per_neuron = int(np.ceil(difference / C_real.shape[0]**2))
            print(number_to_reduce_per_neuron)
        while(difference > 0):
            C_new_discrete[C_new_discrete > 0] -= 1
            C_new_discrete[C_new_discrete < 0] += 1
            
            difference = np.sum(np.abs(C_new_discrete)) - number_available*C_real.shape[0]

        if(debug):
            print(C_new_discrete[:,0])
            total_num_used = np.sum(np.abs(C_new_discrete))
            print("Number used %d / %d" % (total_num_used, C_real.shape[0]*number_available))
    # Stochastic round
    return C_new_discrete



# Obtained from running simulation for 140 iterations
weight_range = (-0.4,0.4)

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

C_real = 0.5*np.random.randn(Nneuron,Nneuron)
delta_C_real = 0.02*np.random.randn(Nneuron,Nneuron)

new_discrete_C = stochastic_round(C_real, delta_C_real, FtM, weight_range)
