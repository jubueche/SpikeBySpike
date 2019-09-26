from brian2 import *
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from utils import Utils
import numpy as np
import os
import sys
import json



def create_network(F, utils, x):

    ########## Input neuron group ##########

    # The input has x and c, since c depends on x: c(t-1)=x(t)-x(t-1)+lambda*delta_t*x(t-1)
    eqs_in = '''
    ct_1 : 1
    xt : 1
    xt_1 : 1
    '''

    I = NeuronGroup(N=utils.n_in, model=eqs_in, method='euler')

    @network_operation(dt=utils.delta_t*ms)
    def update_I(t):
        current_time = int((t/ms)/utils.delta_t) # Normalize row

        I.ct_1_ = int(current_time>0)*x[:,current_time]-x[:,current_time-1]+utils.lambbda*utils.dtt*x[:,current_time-1] #xt-x(t-1)+lambda*dt*x(t-1)
        I.xt_ = x[:,current_time]
        I.xt_1_ = x[:,current_time-1]

    sm_I = StateMonitor(I, variables=True, record=True, dt=utils.delta_t*ms)

    eqs_g = '''
    input : 1
    input_1 : 1
    '''

    G = NeuronGroup(N=utils.N, model = eqs_g, method='euler')

    conn_F = Synapses(I, G, 'weight : 1')

    # Connect fully
    conn_F.connect()

    # Initialize
    conn_F.weight = F.ravel() # NOTE F has shape (utils.n_in,N), => F_{i,j} connects i-th in-neuron to j-th output

    @network_operation(dt=utils.delta_t*ms)
    def update_G(t):
        current_t = int((t/ms)/utils.delta_t) # in [0,duration)

        F_ = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))
        
        ct_1 = np.copy(np.reshape(I.ct_1_, (-1,1)))

        if(current_t == 0):
            input = 0.166*np.reshape(np.asarray(np.random.randn(utils.N)), (-1,1))
        else:
            input = np.matmul(F_.T, ct_1)

        
        # Assign all the local copies to G
        G.input_ = np.copy(np.reshape(input, (-1,)))
        G.input_1_ = np.copy(G.input_)

        
    sm_G = StateMonitor(G, variables=True, record=True, dt=utils.delta_t*ms)
    net = Network(I,sm_I,G, sm_G, update_G, update_I, conn_F)
    net.store('Init')

    return_dict = {
        "net":net,
        "sm_G":sm_G,
        "sm_I":sm_I,
        "conn_F":conn_F
    }

    return return_dict
# End create_network()


#### Script to generate findings in paper "Learning to represent signals spike by spike" [https://arxiv.org/pdf/1703.03777.pdf]
seed(43)
np.set_printoptions(precision=6, suppress=True) # For the rate vector

#! Call Utils constructor with JSON object
utils = Utils.from_default()
utils.penable = True #! Disabled plotting

x = utils.get_matlab_like_input()
plt.figure(figsize=(10,8))
plt.plot(x.T)

if(utils.penable):
    plt.show()

F = np.random.normal(loc=0.0, scale=1.0, size=(utils.n_in, utils.N)) # Initialize F and Omega
for (idx,row) in enumerate(F): # Normalize F
    tmp = utils.gamma* (row / np.sqrt(np.sum(row**2)))
    F[idx,:] = tmp

return_dict = create_network(F, utils, x)

net = return_dict["net"]
sm_G = return_dict["sm_G"]
sm_I = return_dict["sm_I"]
conn_F = return_dict["conn_F"]
    
net.run(duration=utils.duration*ms)

print(sm_G.input_.shape)

for (idx,row) in enumerate(sm_G.input_[1:5,:]) :
    plt.plot(sm_G.input_[idx,1:])

plt.show()

