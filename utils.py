import numpy as np
from scipy.ndimage import gaussian_filter
from moving_average import EMA, MovingAverage
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


class Utils:

    def __init__(self):
        self.colors = ['m','b','y','r','g']
        self.penable = True # Enable plotting
        self.use_learning = True
        self.num_iter = 2 # TODO How many iterations for training?
        self.N = 20
        self.n_in = 2
        self.thresh = 0.5

        self.duration = 8000 #ms # TODO How long should be one run through the signal? Matlab code: 5000 (ms?)
        self.delta_t = 1 # ms
        # TODO correct (2 lines below)?
        self.dtt = 10**-3
        self.lambbda = float(1/50) # TODO Lambda is 50 s^-1

        self.lambbda_f = self.lambbda #Only for first experiment
        self.time_steps = int(self.duration/self.delta_t)
        self.sigma_x = 1 # TODO See table 1 in [https://arxiv.org/src/1703.03777v2/anc/SI.pdf]. In the SI it is 2*10^3
        self.sigma_eps_v = 10**-2 # 10**-3
        self.sigma_eps_t = 2*10**-2
        self.eps_f = 10**-3 #-4 and -3
        self.eps_omega = 10**-2
        self.alpha = 0.21
        self.beta = 1.25
        self.mu = 0.02#0.02 #l2 cost. High l2 cost -> Denser spike trains.

        self.gamma = 40.5#0.8 #Initital FF weight. High -> High initial firing -> High number of updates
        self.omega = -0.5

        self.eta  = 1000 # TODO In the paper it says time constant 6ms. Does that correspond to sigma(std)=6 ?, 600
        self.mA_xrT = MovingAverage(shape=(self.n_in,self.N))
        self.mA_rrT = MovingAverage(shape=(self.N,self.N))
        self.mA_r = MovingAverage(shape=(self.N,1))

    def set_n_in(self, n=2):
        self.n_in = n
        self.mA_xrT = MovingAverage(shape=(self.n_in,self.N))
        self.mA_rrT = MovingAverage(shape=(self.N,self.N))
        self.mA_r = MovingAverage(shape=(self.N,1))

    def reset_averages(self):
        self.mA_xrT.V = np.zeros(shape=self.mA_xrT.V.shape)
        self.mA_rrT.V = np.zeros(shape=self.mA_rrT.V.shape)
        self.mA_r.V = np.zeros(shape=self.mA_r.V.shape)
        self.mA_xrT.n = 0
        self.mA_rrT.n = 0
        self.mA_r.n = 0
    
    def get_input(self):
        xx = np.random.normal(loc=0.0, scale=self.sigma_x**2, size=(self.n_in, self.time_steps))
        x = gaussian_filter(xx[0,:], sigma=self.eta)
        for i in range(1,self.n_in):
            x = np.vstack([x, gaussian_filter(xx[i,:], sigma=self.eta)])
        
        return x

    def get_matlab_like_input(self):
        T = self.duration
        dim = self.n_in
        seed = np.random.randn(dim, T)*self.sigma_x
        L = round(6*self.eta)
        cNS = np.hstack([np.zeros((dim,1)), seed])

        ker = np.exp( -((np.linspace(1,L,L) - L/2))**2/self.eta**2)
        ker = ker/sum(ker)

        x = np.zeros((dim, max([cNS.shape[1]+len(ker)-1,len(ker),cNS.shape[1] ])))

        for i in range(0,dim):
            x[i,:] = np.convolve(cNS[i,:], ker)*np.sqrt(self.eta/0.4)

        x = x[:, 0:T]
        return x


    def get_mixed_input(self, low, high):
        xx = np.random.normal(loc=0.0, scale=self.sigma_x**2, size=(self.n_in, self.time_steps))
        eta = np.random.randint(low=low, high=high)
        x = gaussian_filter(xx[0,:], sigma=eta)
        for i in range(1,self.n_in):
            eta = np.random.randint(low=low, high=high)
            x = np.vstack([x, gaussian_filter(xx[i,:], sigma=eta)])
        
        return x

    def get_mixed_input_reoc(self, low=200, high=1000, n=1):
        x = self.get_mixed_input(low, high)
        self.duration = self.duration*n
        self.time_steps = int(self.duration/self.delta_t)
        xx = x
        for i in range(1,n):
            xx = np.hstack([xx,x])

        return xx


    def get_decoder(self):
        D = np.matmul(self.mA_xrT.get_value(), np.linalg.pinv(self.mA_rrT.get_value()))
        return D

    def get_true_decoder(self, F, Omega):
        D = np.matmul(-np.linalg.pinv(F), Omega)
        return D


    # -1 means that we compute the error over all the points recorded so far
    # Else compute the error over the last tt points
    def get_error(self, rate, xt, until = -1):

        assert self.mA_xrT.get_value().shape == (self.n_in, self.N), "Err: Shape must have dimesnion (utils.n_in, N)"
        assert self.mA_rrT.get_value().shape == (self.N,self.N), "Err: Shape must have dimension (N,N)"
        errors = []
        D = self.get_decoder()
        if(until == -1):
            x_hat = np.matmul(D, rate)
            x = xt

            
            for i in range(x.shape[0]):
                errors.append(np.linalg.norm(x_hat[i,:] - x[i,:], 2))
            
        return (errors,x.T,x_hat.T, D)

    def reconstruction_error_over_time(self, x, x_hat, dt = 50):

        err0 = []
        err1 = []
        for i in range(0, x.shape[0], dt):
                tmp_x0 = x[i:i+dt,0]
                tmp_x_hat0 = x_hat[i:i+dt,0]
                err0.append(np.linalg.norm(tmp_x0-tmp_x_hat0, 2))
                tmp_x1 = x[i:i+dt,1]
                tmp_x_hat1 = x_hat[i:i+dt,1] 
                err1.append(np.linalg.norm(tmp_x1-tmp_x_hat1, 2))

        return (np.asarray(err0), np.asarray(err1))

    def reconstruction_error_over_time_list(self, x, x_hat, dt = 50):
        errors = []

        for i in range(0,x.shape[1]):
            err_tmp = []
            
            for j in range(0, x.shape[0], dt):
                tmp_x0 = x[j:j+dt,i]
                tmp_x_hat0 = x_hat[j:j+dt,i]
                err_tmp.append(np.linalg.norm(tmp_x0-tmp_x_hat0, 2))
            errors.append(err_tmp)
        return errors


    def plot_results(self, x_hat_first, times_first, indices_first, x, x_hat, times, indices, errors, num_spikes):
        if(self.penable):
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

                ps_first = []
                for i in range(self.n_in):
                        title = ("Initial Reconstruction of x%d" % i)
                        ps_first.append(win.addPlot(title=title))
                        win.nextRow()

                p_initial_spikes = win.addPlot(title="Initial Spikes")
                win.nextRow()

                # Uncomment for different plot for each dimension of x
                ps = []
                for i in range(self.n_in):
                        title = ("Reconstruction of x%d" % i)
                        ps.append(win.addPlot(title=title))
                        win.nextRow()

                p_after_spikes = win.addPlot(title="Spikes")
                win.nextRow()
                p_recon_error = win.addPlot(title="Reconstruction error over time")
                win.nextRow()
                p_sparsity = win.addPlot(title="Sparsity (Number of spikes)")
                win.nextRow()

                # Uncomment for different plot for each dimension of x
                for i in range(self.n_in):
                        ps[i].plot(y=x[:,i], pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))
                        ps[i].plot(y=x_hat[:,i], pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.DashLine))


                p_initial_spikes.plot(x=times_first, y=indices_first,
                        pen=None, symbol='o', symbolPen=None,
                        symbolSize=3, symbolBrush=(68, 245, 255))

                for i in range(self.n_in):
                    ps_first[i].plot(y=x[:,i], pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))
                    ps_first[i].plot(y=x_hat_first[:,i], pen=pg.mkPen('g', width=1, style=pg.QtCore.Qt.DashLine))

                p_after_spikes.plot(x=times, y=indices,
                                pen=None, symbol='o', symbolPen=None,
                                symbolSize=3, symbolBrush=(68, 245, 255))

                
                for idx in range(0,errors.shape[1]):
                        p_recon_error.plot(y=errors[:,idx], pen=pg.mkPen(self.colors[idx], width=1, style=pg.QtCore.Qt.DashLine))

                p_sparsity.plot(y=num_spikes, pen=pg.mkPen('y', width=1, style=pg.QtCore.Qt.DashLine))

                app.exec()