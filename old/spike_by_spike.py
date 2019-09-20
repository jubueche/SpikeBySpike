from brian2 import *
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
from utils import Utils

utils = Utils()

#### Script to generate findings in paper "Learning to represent signals spike by spike" [https://arxiv.org/pdf/1703.03777.pdf]
seed(42)
x = utils.get_input()


# Voltage noise
eps_v = np.random.normal(loc=0.0, scale=utils.sigma_eps_v, size=(utils.N, utils.time_steps))

# Threshold 'noise'
eps_t = np.random.normal(loc=0.0, scale=utils.sigma_eps_t, size=(utils.N, utils.time_steps))

# Initialize F and Omega
F = np.random.normal(loc=0.0, scale=1.0, size=(utils.n_in, utils.N))
Omega = np.eye(utils.N,utils.N)*utils.omega
# Normalize rows
for (idx,row) in enumerate(F):
    tmp = utils.gamma* (row / np.sqrt(np.sum(row**2)))
    F[idx,:] = tmp

### Input neuron group

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
conn_F.weight = F.reshape((-1,)) # NOTE F has shape (utils.n_in,N), => F_{i,j} connects i-th in-neuron to j-th output
conn_Omega.weight = Omega.reshape((-1,)) # NOTE Omega has shape (N,N) so no information is gained here.  We are using the paper version.


@network_operation(dt=utils.delta_t*ms)
def update_G(t):
    current_t = int((t/ms)/utils.delta_t) # in [0,duration)

    F_ = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))
    Omega_ = np.copy(np.reshape(conn_Omega.weight, (utils.N, utils.N)))
    
    ct_1 = np.copy(np.reshape(I.ct_1_, (-1,1)))
    vt_1 = np.copy(np.reshape(G.vt_1, (-1,1)))
    ot_1 = np.copy(np.reshape(G.ot_1, (-1,1)))
    rt_1 = np.copy(np.reshape(G.rt_1, (-1,1)))

    vt = int(current_t>0)*((1-utils.lambbda*utils.dtt)*vt_1 + utils.dtt*np.matmul(F_.T, ct_1) + np.matmul(Omega_, ot_1) + np.reshape(eps_v[:, current_t], (-1,1)))
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
        if((G.active_ == 1).all()):
                F_[:,k] = np.reshape(np.reshape(F_[:,k], (-1,1)) + utils.eps_f*(utils.alpha*np.reshape(I.xt_1_,(-1,1)) - np.reshape(F_[:,k], (-1,1))), (-1,))
                tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(vt_1 + utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,))
                #tmp = np.reshape(np.reshape(Omega_[:,k], (-1,1)) - utils.eps_omega*(utils.beta*(utils.mu*rt_1) + np.reshape(Omega_[:,k], (-1,1))), (-1,)) # DOnt use vt
                tmp1 = Omega_[k,k] - utils.eps_omega*utils.mu
                Omega_[:,k] = tmp
                Omega_[k,k] = tmp1

        #print("Delta F: %.4f" % np.linalg.norm(F_ - np.reshape(conn_F.weight, F_.shape)))
        #print(np.reshape(conn_F.weight, F_.shape))

        # Assign
        if((G.active_ == 1).all()):
                conn_F.weight = np.copy(np.reshape(F_, (-1,)))
                conn_Omega.weight = np.copy(np.reshape(Omega_, (-1,)))


    rt = (1-utils.lambbda*utils.dtt)*rt_1 + ot_1

    # Assign all the local copies to G
    G.vt_ = np.copy(np.reshape(vt, (-1,)))
    G.vt_1_ = np.copy(G.vt_)

    G.ot_ = np.copy(np.reshape(ot, (-1,)))
    G.ot_1_ = np.copy(G.ot_)


    G.rt_ = np.copy(np.reshape(rt, (-1,)))
    G.rt_1_ = np.copy(G.rt_) 
    
    # Update moving averages
    utils.mA_r.update(rt)
    utils.mA_rrT.update(np.matmul(rt,rt.T))
    utils.mA_xrT.update(np.matmul(np.reshape(I.xt_, (-1,1)),rt.T))

    D = np.matmul(utils.mA_xrT.get_value(), np.linalg.pinv(utils.mA_rrT.get_value()))

    


sm_G = StateMonitor(G, variables=True, record=True, dt=utils.delta_t*ms)
net = Network(I,sm_I,G, sm_G, update_G, update_I, conn_F, conn_Omega)
net.store('Init')
### Training
G.active = 0
net.run(duration=utils.duration*ms)


# Error
(err0, err1, _, x_hat, _) = utils.get_error(sm_G.rt_, sm_I.xt_)
print("Err0: %.6f    Err1: %.12f" % (err0, err1))

# The trained weights, important: Need to copy because otherwise it will reference conn_X.weight, which changes when restore is called.
Omega_after_training = np.copy(np.reshape(conn_Omega.weight, (utils.N,utils.N)))
F_after_training = np.copy(np.reshape(conn_F.weight, (utils.n_in, utils.N)))

net.restore('Init')

# Reset the running averages
utils.reset_averages()


conn_Omega.weight = np.copy(np.reshape(Omega_after_training, (-1,)))
conn_F.weight = np.copy(np.reshape(F_after_training, (-1,)))

G.active = 1
net.run(duration = utils.duration*ms)

Omega_after_test = np.reshape(conn_Omega.weight, (utils.N,utils.N))
F_after_test = np.reshape(conn_F.weight, (utils.n_in, utils.N))

'''assert np.linalg.norm(Omega_after_training-Omega_after_test) == 0, "Err: Testing changed the weights"
assert np.linalg.norm(F_after_training-F_after_test) == 0, "Err: Testing changed the weights"
'''

(err0, err1, x, x_hat, D) = utils.get_error(sm_G.rt_, sm_I.xt_)
print("Err0: %.6f    Err1: %.12f" % (err0, err1))

# The spike trains
ot = sm_G.ot
indices, times = np.where(ot == 1)

err0,err1 = utils.reconstruction_error_over_time(x,x_hat,dt=100)


### Plotting ###
penable = True
if(penable):
        app = QtGui.QApplication.instance()
        if app is None:
                app = QtGui.QApplication(sys.argv)
        else:
                print('QApplication instance already exists: %s' % str(app))

        pg.setConfigOptions(antialias=True)
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        win = pg.GraphicsWindow()
        win.resize(1500, 1500)
        win.setWindowTitle('Learning to represent signals spike-by-spike')
        p1 = win.addPlot(title="Reconstruction of x0")
        win.nextRow()
        p2 = win.addPlot(title="Reconstructon of x1")
        win.nextRow()
        p3 = win.addPlot(title="Spikes")
        win.nextRow()
        p4 = win.addPlot(title="Reconstruction error over time")
        win.nextRow()

        p1.plot(y=x[:,0], pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))
        p1.plot(y=x_hat[:,0], pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.DashLine))

        p2.plot(y=x[:,1], pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))
        p2.plot(y=x_hat[:,1], pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.DashLine))

        p3.plot(x=times, y=indices,
                        pen=None, symbol='o', symbolPen=None,
                        symbolSize=3, symbolBrush=(68, 245, 255))

        p4.plot(y=err0, pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))
        p4.plot(y=err1, pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.DashLine))
        app.exec()