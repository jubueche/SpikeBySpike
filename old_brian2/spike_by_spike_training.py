from brian2 import *
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from utils import Utils
import numpy as np
import os
import sys
import json

########## Parse training and testing input JSON files ##########
if(sys.argv[1] != None):
    saveFolder = sys.argv[1]
else:
    raise SystemExit('Error: Please specify folder to save data in.')
if(sys.argv[2] != None):
    with open(os.path.join(os.getcwd(), sys.argv[2]), 'r') as f:
        parameters = json.load(f)
else:
    raise SystemExit('Error: Please add parameters file.')



def create_network(F, Omega, utils, x):
    eps_v = np.random.normal(loc=0.0, scale=utils.sigma_eps_v, size=(utils.N, utils.time_steps)) # Voltage noise
    eps_t = np.random.normal(loc=0.0, scale=utils.sigma_eps_t, size=(utils.N, utils.time_steps)) # Threshold 'noise'

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
    v_recon : 1
    vt : 1
    vt_1 : 1
    ot : 1
    ot_1 : 1
    rt : 1
    rt_1 : 1
    active : 1
    '''

    G = NeuronGroup(N=utils.N, model = eqs_g, method='euler')

    conn_F = Synapses(I, G, 'weight : 1')
    conn_Omega = Synapses(G, G, 'weight : 1')

    # Connect fully
    conn_F.connect()
    conn_Omega.connect()

    # Initialize
    conn_F.weight = F.ravel() # NOTE F has shape (utils.n_in,N), => F_{i,j} connects i-th in-neuron to j-th output
    conn_Omega.weight = Omega.ravel() # NOTE Omega has shape (N,N) so no information is gained here.  We are using the paper version.

    Omega_offline = Omega # The offline version of Omega. Omega is always updated in the learning rule, but a thresholded version is used for the network.

    @network_operation(dt=utils.delta_t*ms)
    def update_G(t):


        current_t = int((t/ms)/utils.delta_t) # in [0,duration)

        F_ = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))
        Omega_ = np.copy(np.reshape(conn_Omega.weight, (utils.N, utils.N)))

        ct_1 = np.copy(np.reshape(I.ct_1_, (-1,1)))
        vt_1 = np.copy(np.reshape(G.vt_1, (-1,1)))
        ot_1 = np.copy(np.reshape(G.ot_1, (-1,1)))
        rt_1 = np.copy(np.reshape(G.rt_1, (-1,1)))


        # Compute reconstructed voltage using x and r_t
        # Shapes: D:  (#x,N) , x: (#x, Num_data_points), r_t: (N,1)
        # Formula is V_n(t) = D_n^T*x - D_n^T*Dr
        x_t = np.reshape(x[:, current_t], (-1,1))
        r_tmp = np.reshape(rt_1, (-1,1))

        old_voltage_reconstructed = G.v_recon_    
        voltage_reconstructed = np.matmul(F_.T, x_t) + np.matmul(Omega_,r_tmp)


        if(current_t == 0):
            vt = 0.166*np.reshape(np.asarray(np.random.randn(utils.N)), (-1,1))
        else:
            vt = ((1-utils.lambbda*utils.dtt)*vt_1 + utils.dtt*np.matmul(F_.T, ct_1) + np.matmul(Omega_, ot_1) + np.reshape(eps_v[:, current_t], (-1,1)))

        ot = np.zeros(shape=ot_1.shape)
        
        T = utils.thresh*np.ones(shape=(utils.N,1)) - np.reshape(eps_t[:,current_t],(-1,1))
        k = np.argmax(vt - T)

        # Reset the integration error, because we know the level of the threshold
        for i in range(len(old_voltage_reconstructed)):
            diff = old_voltage_reconstructed[i] - voltage_reconstructed[i]
            if(diff > T[i]-0.05):
                voltage_reconstructed[i] = T[i] - diff

        #print(vt[n])

        # Rule in the paper is Delta Omega_{n,k} = -beta(V_n + mu*r_n) - Omega_{n,k} - ... if neuron k spikes
        # Omega_{n,k} means the connection from neuron k to neuron n.
        # Omega_{:,k} = Omega_{k} (the k-th column of Omega) are all the weights of the connections from k -> All others including k itself
        # When neuron n receives the spike, it only has V_n, r_n at hand (locality). For the update it only needs exactly that.
        # Neuron n updates the incoming connection k -> n, which is denoted by Omega_{n,k}. From the update rule one can see that this requires only
        # local information.
        # As a vector notation, the update can be done using:
        # Delta Omega_{k} = -beta*(V + mu*r) - Omega_{k} - ...if neuron k spikes
        # We discussed that this is not local, but it is:
        #           | Omega_{1,k} |          | V_1 |       | r_1 |    | Omega_{1,k} |
        #           |      ...    |          | ... |       | ... |    |      ...    |
        #   Delta   | Omega_{n,k} | = -beta*(| V_n | + mu* | r_n |) - | Omega_{n,k} |
        #           |      ...    |          | ... |       | ... |    |      ...    |
        #           | Omega_{N,k} |          | V_N |       | r_N |    | Omega_{N,k} |


        if(vt[k] > T[k]):
            ot[k] = 1
            if(utils.use_learning):
                F_[:,k] = np.reshape(np.reshape(F_[:,k], (-1,1)) + utils.eps_f*(utils.alpha*np.reshape(I.xt_1_,(-1,1)) - np.reshape(F_[:,k], (-1,1))), (-1,))
                #tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(vt_1 + utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,))
                #tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,)) # DOnt use vt
                # Use the reconstructed voltage in the update. Use G.v_recon_ since at this point it refers to v(t-1). voltage_reconstructed holds the value of v(t). The update is performed at the bottom.
                # tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(np.reshape(G.v_recon_, (-1,1)) + utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,)) #! No Omega offline
                tmp = np.reshape(np.reshape(Omega_offline[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(np.reshape(G.v_recon_, (-1,1)) + utils.mu*rt_1) + np.reshape(Omega_offline[:,k], (-1,1))), (-1,))
                # tmp1 = Omega_[k,k] - utils.eps_omega*utils.mu #! No single threshold update on DYNAP-SE
                #update_threshold = np.ones(utils.N)*Omega_.diagonal() - utils.eps_omega*utils.mu*0.1 #! No Omega offline
                update_threshold = np.ones(utils.N)*Omega_offline.diagonal() - utils.eps_omega*utils.mu*0.1

                tmp[abs(tmp) < utils.cutoff] = 0
                #Omega_[:,k] = tmp #! No Omega offline
                Omega_offline[:,k] = tmp
                #Omega_[k,k] = tmp1 #! No single threshold update on DYNAP-SE
                # np.fill_diagonal(Omega_, update_threshold) # On DYNAPS, can only update all thresholds #! No Omega offline
                np.fill_diagonal(Omega_offline, update_threshold) # On DYNAPS, can only update all thresholds
                
                # Assign
                # conn_F.weight = np.copy(np.reshape(F_, (-1,))) #! No F update on DYNAP-SE
                # conn_Omega.weight = np.copy(np.reshape(Omega_, (-1,))) #! No Omega offline
                conn_Omega.weight = np.copy(np.reshape(utils.threshold_matrix(Omega_offline), (-1,))) # ! Need to threshold the offline version here

        rt = (1-utils.lambbda*utils.dtt)*rt_1 + ot_1
        #print(rt)

        # Assign all the local copies to G
        G.vt_ = np.copy(np.reshape(vt, (-1,)))
        G.vt_1_ = np.copy(G.vt_)

        G.ot_ = np.copy(np.reshape(ot, (-1,)))
        G.ot_1_ = np.copy(G.ot_)

        G.rt_ = np.copy(np.reshape(rt, (-1,)))
        G.rt_1_ = np.copy(G.rt_) 

        G.v_recon_ = np.copy(np.reshape(voltage_reconstructed, (-1)))
        
        # Update moving averages
        utils.mA_r.update(rt)
        utils.mA_rrT.update(np.matmul(rt,rt.T))
        utils.mA_xrT.update(np.matmul(np.reshape(I.xt_, (-1,1)),rt.T))

        
    sm_G = StateMonitor(G, variables=True, record=True, dt=utils.delta_t*ms)
    net = Network(I,sm_I,G, sm_G, update_G, update_I, conn_F, conn_Omega)
    net.store('Init')

    return_dict = {
        "net":net,
        "sm_G":sm_G,
        "sm_I":sm_I,
        "conn_Omega":conn_Omega,
        "conn_F":conn_F
    }

    return return_dict
# End create_network()

########### Create a new folder storing plots ###########
if saveFolder:
    direc = os.path.join(os.getcwd(), saveFolder)
else:
    direc = os.path.join(os.getcwd(), "log")

direc_training = os.path.join(direc, "training")
direc_testing = os.path.join(direc, "testing")
if(not os.path.exists(direc)):
    os.mkdir(direc)
if(not os.path.exists(direc_training)):
    os.mkdir(direc_training)
if(not os.path.exists(direc_testing)):
    os.mkdir(direc_testing)

########### Create the network ###########

#### Script to generate findings in paper "Learning to represent signals spike by spike" [https://arxiv.org/pdf/1703.03777.pdf]
seed(43)
np.set_printoptions(precision=6, suppress=True) # For the rate vector

#! Call Utils constructor with JSON object
utils = Utils.from_json(parameters["training"])

x = utils.get_matlab_like_input()
plt.figure(figsize=(10,8))
plt.plot(x.T)
plt.savefig(os.path.join(direc_training, "training_input.png"))
if(utils.penable):
    plt.show()

F = np.random.normal(loc=0.0, scale=1.0, size=(utils.n_in, utils.N)) # Initialize F and Omega
for (idx,row) in enumerate(F): # Normalize F
    tmp = utils.gamma* (row / np.sqrt(np.sum(row**2)))
    F[idx,:] = tmp

# Pickle the feed forward matrix
F.dump(os.path.join("DYNAPS/Resources", "F.dat"))

Omega = np.eye(utils.N,utils.N)*utils.omega # Initialize Omega

return_dict = create_network(F, Omega, utils, x)

net = return_dict["net"]
sm_G = return_dict["sm_G"]
sm_I = return_dict["sm_I"]
conn_Omega = return_dict["conn_Omega"]
conn_F = return_dict["conn_F"]

x_hat_first = np.ones(shape=(utils.time_steps, utils.n_in))

# Save the matrix before training
utils.save_omega(os.path.join(direc_training, "omega_heat_map_before_training.png"), Omega)

########### Training ###########
log_file = open(os.path.join(direc_training, 'log.txt'), "w+")

print("Training...")
log_file.write("Training...\r\n")

errors = []
num_spikes = []
for i in range(0,utils.num_iter):

    if(i == 0 or i == utils.num_iter-1):
        utils.use_learning = False
    else:
        utils.use_learning = True

    Omega_before = np.copy(np.reshape(conn_Omega.weight, (utils.N,utils.N)))
    F_before = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))
    
    net.run(duration=utils.duration*ms)

    Omega_after = np.copy(np.reshape(conn_Omega.weight, (utils.N,utils.N)))
    F_after = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))

    # utils.save_omega(os.path.join(direc_training, ("omega_heat_map_iter%d.png" % i)), Omega_after) #! Uncomment for storing Omega heat map at every iter

    delta_F = np.linalg.norm(F_before.ravel()-F_after.ravel(), 2)
    delta_Omega = np.linalg.norm(Omega_before.ravel()-Omega_after.ravel(), 2)

    # The spike trains
    ot = sm_G.ot
    indices, times = np.where(ot == 1)

    # Reconstructed voltages
    v_recon = sm_G.v_recon
    v_true = sm_G.vt

    # Error
    (errs, _, x_hat, D) = utils.get_error(sm_G.rt_, sm_I.xt_)
    errors.append(errs)
    s = ""
    for j in range(len(errs)):
        s = s + ("Err%d is " % j) + ("%.6f" % errs[j]) + "    "
    
    s = s + ("#Spikes: %d" % np.sum(ot.ravel()))
    s = s + ("    Delta F: %.6f    Delta Omega: %.6f" % (delta_F, delta_Omega))
    print(s)
    log_file.write(s + "\r\n")

    ##### Decay learning rate #####
    if((i+1) % 5 == 0):
        utils.eps_omega = utils.eps_omega *0.75 # was 0.5
        utils.eps_f = utils.eps_f *0.75
        s = ("Reduced learning rate:     Eps Omega: %.6f     Eps F: %.6f" % (utils.eps_omega, utils.eps_f))
        print(s)
        log_file.write(s + "\r\n")
    num_spikes.append(np.sum(ot.ravel()))

    # Collect x_hat from the first run w/o training
    if(i==0):
        x_hat_first = x_hat
        indices_first, times_first = np.where(ot == 1)

    # Reset running averages for the decoder
    utils.reset_averages()

    net.restore(name='Init')
    conn_F.weight = np.reshape(F_after, (-1,))
    conn_Omega.weight = np.reshape(Omega_after, (-1,))

log_file.close()

x = x.T
errors = np.asarray(errors)

n = 5
plt.figure(figsize=(10,8))
for i in range(0,n):
        plt.plot(v_recon[i,:], utils.colors[i], label="Reconstructed voltage")
        plt.plot(v_true[i,:], utils.colors[i], label="True voltage")
plt.legend()
plt.savefig(os.path.join(direc_training, "vt_vs_reconstructed_vt.png"))
if(utils.penable):
    plt.show()

########### Plotting ###########
utils.plot_results(x_hat_first, times_first, indices_first, x, x_hat, times, indices, errors, num_spikes, os.path.join(direc_training, "after_training.png"))


########### Testing ###########
print("Testing...")
Omega = Omega_after
F = F_after

########## Plot distribution of weights ##########
weights = Omega_after.ravel()
plt.figure(figsize=(10,8))
plt.hist(weights, 50, density=True, facecolor='green', alpha=0.75)
plt.savefig(os.path.join(direc_training, "omega_weight_distribution_post_learning.png"))
if(utils.penable):
    plt.show() 

# Number of weights different from 0
Omega_after.dump(os.path.join("DYNAPS/Resources", "Omega_after.dat"))

# Save the matrix after training
utils.save_omega(os.path.join(direc_training, "omega_heat_map_after_training.png"), Omega)


num_signals = 2

for k in range(num_signals):
    utils_testing = Utils.from_json(parameters["testing"])
    utils.use_learning = False
    x_testing = utils_testing.get_matlab_like_input()
    plt.figure(figsize=(10,8))
    plt.plot(x_testing.T)
    plt.savefig(os.path.join(direc_testing, (("test_signal_%d.png") % k)))
    if(utils_testing.penable):
        plt.show()

    return_dict_testing = create_network(F, Omega, utils_testing, x_testing)

    net = return_dict_testing["net"]
    sm_G = return_dict_testing["sm_G"]
    sm_I = return_dict_testing["sm_I"]
    conn_Omega = return_dict_testing["conn_Omega"]
    conn_F = return_dict_testing["conn_F"]

    net.run(utils_testing.duration*ms)

    # The spike trains
    ot = sm_G.ot
    indices, times = np.where(ot == 1)

    (errs, _, x_hat, D) = utils_testing.get_error(sm_G.rt_, sm_I.xt_)

    ########### Plotting testing ###########
    utils_testing.plot_test_signal(x_hat, x_testing, indices, times, os.path.join(direc_testing, ("test_signal_reconstructed%d.png" % k)))