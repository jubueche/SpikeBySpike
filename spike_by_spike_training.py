from brian2 import *
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from utils import Utils
import numpy as np

utils = Utils()

#### Script to generate findings in paper "Learning to represent signals spike by spike" [https://arxiv.org/pdf/1703.03777.pdf]
seed(42)
np.set_printoptions(precision=6, suppress=True) # For the rate vector

x = utils.get_matlab_like_input()
plt.plot(x.T)
plt.show()

eps_v = np.random.normal(loc=0.0, scale=utils.sigma_eps_v, size=(utils.N, utils.time_steps)) # Voltage noise
eps_t = np.random.normal(loc=0.0, scale=utils.sigma_eps_t, size=(utils.N, utils.time_steps)) # Threshold 'noise'

F = np.random.normal(loc=0.0, scale=1.0, size=(utils.n_in, utils.N)) # Initialize F and Omega
for (idx,row) in enumerate(F): # Normalize F
    tmp = utils.gamma* (row / np.sqrt(np.sum(row**2)))
    F[idx,:] = tmp
Omega = np.eye(utils.N,utils.N)*utils.omega # Initialize Omega

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
    # Forumula is V_n(t) = D_n^T*x - D_n^T*Dr
    x_t = np.reshape(x[:, current_t], (-1,1))
    r_tmp = np.reshape(rt_1, (-1,1))
    voltage_reconstructed = np.matmul(F_.T, x_t) + np.matmul(Omega_,r_tmp)

    if(current_t == 0):
        vt = 0.166*np.reshape(np.asarray(np.random.randn(utils.N)), (-1,1))
    else:
        vt = ((1-utils.lambbda*utils.dtt)*vt_1 + np.matmul(F_.T, ct_1) + np.matmul(Omega_, ot_1) + np.reshape(eps_v[:, current_t], (-1,1)))

    ot = np.zeros(shape=ot_1.shape)
    
    # TODO Check this. This is different in the paper
    T = utils.thresh*np.ones(shape=(utils.N,1)) - np.reshape(eps_t[:,current_t],(-1,1))
    k = np.argmax(vt - T)
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
            tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(vt_1 + utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,))
            #tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,)) # DOnt use vt
            # Use the reconstructed voltage in the update. Use G.v_recon_ since at this point it refers to v(t-1). voltage_reconstructed holds the value of v(t). The update is performed at the bottom.
            #tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(np.reshape(G.v_recon_, (-1,1)) + utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,))
            tmp1 = Omega_[k,k] - utils.eps_omega*utils.mu
            Omega_[:,k] = tmp
            Omega_[k,k] = tmp1

            # Assign
            conn_F.weight = np.copy(np.reshape(F_, (-1,)))
            conn_Omega.weight = np.copy(np.reshape(Omega_, (-1,)))


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

x_hat_first = np.ones(shape=(utils.time_steps, utils.n_in))

########## Training ##########
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
    
    s = s + ("    #Spikes: %d" % np.sum(ot.ravel()))
    s = s + ("    Delta F: %.6f    Delta Omega: %.6f" % (delta_F, delta_Omega))
    utils.eps_omega = utils.eps_omega / 2
    utils.eps_f = utils.eps_f / 2
    print(s)
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

x = x.T
errors = np.asarray(errors)

n = 5
for i in range(0,n):
        #plt.plot(v_recon[i,:], colors[i], label="Reconstructed voltage")
        plt.plot(v_true[i,:], utils.colors[i], label="True voltage")
plt.legend()
plt.show()


### Plotting ###
utils.plot_results(x_hat_first, times_first, indices_first, x, x_hat, times, indices, errors, num_spikes)