import numpy as np
from scipy.ndimage import gaussian_filter
from moving_average import EMA, MovingAverage


class Utils:

    def __init__(self):
        self.use_learning = True
        self.num_iter = 20 # TODO How many iterations for training?
        self.N = 20
        self.n_in = 2
        self.thresh = 0.5

        self.duration = 1000 #ms # TODO How long should be one run through the signal? Matlab code: 5000 (ms?)
        self.delta_t = 1 # ms
        # TODO correct (2 lines below)?
        self.dtt = 10**-3
        self.lambbda = float(1/50) # TODO Lambda is 50 s^-1

        self.lambbda_f = self.lambbda #Only for first experiment
        self.time_steps = int(self.duration/self.delta_t)
        self.sigma_x = 2*10**1 # TODO See table 1 in [https://arxiv.org/src/1703.03777v2/anc/SI.pdf]. In the SI it is 2*10^3
        self.sigma_eps_v = 10**-3
        self.sigma_eps_t = 2*10**-2
        self.eps_f = 10**-4
        self.eps_omega = 10**-3
        self.alpha = 0.21
        self.beta = 1.25
        self.mu = 0.02

        self.gamma = 2.5#0-8
        self.omega = -0.5

        self.eta  = 100 # TODO In the paper it says time constant 6ms. Does that correspond to sigma(std)=6 ?, 600
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
