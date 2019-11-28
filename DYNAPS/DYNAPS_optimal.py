# $ exec(open("./start_rpyc.py").read())

import sbs_dynapse_controller
import argparse
import sys
import os
import argparse
sys.path.append(os.path.join(os.getcwd(),"../"))
from Utils import Utils
import numpy as np
import traceback
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training of optimal spike based signal representation on a neuromorphic chip.')
parser.add_argument('-cc', help='clear the CAMs on the chip',  action='store_true')
parser.add_argument('-d', help='debug mode on',  action='store_true')

args = vars(parser.parse_args())
clear_cam = args['cc']
debug = args['d']

def clean_up_conn(sbs):
    for n in sbs.population:
        sbs.connector.remove_receiving_connections(n)

    sbs.c.close()

def get_input(duration, utils, w):

    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), duration)).T
    Input[:,0:100] = 0
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

    return Input

self_path = sbs_dynapse_controller.__file__
self_path = self_path[0:self_path.rfind('/')]

sbs = sbs_dynapse_controller.SBSController.from_default(clear_cam = clear_cam, debug=debug, num_signals=2)
print("Setting DYNAPS parameters...")
sbs.c.execute("exec(open('" + self_path + "/Resources/DYNAPS/balanced.py').read())")
sbs.model.apply_diff_state()

utils = Utils(Nneuron=10, Nx=2, lam=50, dt=0.001, epsr=0.001, epsf=0.0001, alpha=0.18, beta=1.11, mu=0.022,
            gamma=1.0, Thresh=0.5, Nit=140, Ntime=1000, A=2000, sigma=30,
            dynapse_maximal_synapse_ff=8, dynapse_maximal_synapse_o=2, alignment_delta_t=10)

w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
w = w / np.sum(w)

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

try:
    # Set the excitatory and inh. weights on the DYNAPS
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 255, 7)
    #sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 255, 7)

    delta_mod_thresh_up = 0.05 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.05

    M = np.asarray([[1,-1,0,0],[0,0,1,-1]])
    angles = np.linspace(0,2*np.pi,num=utils.Nneuron)
    D = np.vstack((np.sin(angles),np.cos(angles)))
    """plt.plot(D[0,:],D[1,:], 'o')
    plt.grid(True)
    plt.show()
    print(np.linalg.norm(D,axis=0,ord=2))
    D = np.random.randn(utils.Nx,utils.Nneuron)
    # Normalize rows of D
    D /= np.linalg.norm(D,ord=2,axis=0)"""
    D = np.round(2.6*D).astype(np.int)
    Omega = -np.matmul(D.T,D)
    F = D.T
    FM = np.matmul(F,M)
    print(np.sum(np.abs(FM),axis=1) + np.sum(np.abs(Omega), axis=1))
    assert ((np.sum(np.abs(FM),axis=1) + np.sum(np.abs(Omega), axis=1)) < 60).all(), "Weights exceed max. fan-in"

    # Set the weights on the chip
    sbs.set_feedforward_connection(FM)
    sbs.set_recurrent_weight_directly(Omega)
    sbs.set_recurrent_connection()

    Input = get_input(utils.Ntime, utils, w)
    X = np.zeros((utils.Nx,utils.Ntime))
    for t in range(1, utils.Ntime):
        X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:, t]

    sbs.load_signal(np.copy(X), delta_mod_thresh_up, delta_mod_thresh_dwn)
    O_DYNAPS = sbs.execute()

    plt.figure(figsize=(12,6))
    coordinates = np.nonzero(O_DYNAPS)
    plt.subplot(211)
    plt.scatter(coordinates[1], coordinates[0], s=0.1, marker='o', c='b')
    plt.ylim([0,utils.Nneuron])
    

    R = np.zeros((utils.Nneuron,utils.Ntime))
    for t in range(1,utils.Ntime):
        R[:,t] = (1-utils.lam*utils.dt)*R[:,t-1] + O_DYNAPS[:,t]

    plt.subplot(212)
    x_hat = np.matmul(D,R)
    plt.plot(x_hat[0,:])
    plt.plot(X[0,:])
    plt.show()

except Exception:
    traceback.print_exc()
    clean_up_conn(sbs)
    print("Exception! Cleaned up connections")
except KeyboardInterrupt:
    clean_up_conn(sbs)
    print("Interrupt. Cleaned up connections.")
else:
    clean_up_conn(sbs)    
    print("Cleaned up connections")
