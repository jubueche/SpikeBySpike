import numpy as np
import matplotlib.pyplot as plt
from Utils import my_max, ups_downs_to_O
from runnet import *
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DYNAPS/"))
from helper import signal_to_spike_refractory


def spiking_to_continous(utils):

    # Generate input
    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)
    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

    # Convert the input to the target
    x = np.zeros(Input.shape)
    for t in range(1,utils.Ntime):
        x[:,t] = (1-utils.lam*utils.dt)*x[:,t-1] + utils.dt*Input[:,t]

    # Generate spikes from continous input
    #lams = [10**-i for i in range(5)]
    lams = [0.0, 0.01, 0.001, 0.0001, 0.00001]
    #Rs = [10**-i for i in range(5)]
    Rs = [1.0, 0.9999, 0.999, 0.99, 0.9]
    threshs = np.linspace(0.5, 80, 300)
    old_err = -1
    for lam in lams:
        for R in Rs:
            for thresh in threshs:
                err = run_trial(utils, Input, x, thresh, R, lam)
                if(old_err == -1 or err < old_err):
                    print(err)
                    old_err = err
                    b_lam = lam; b_R = R; b_thresh = thresh

    print(("Best lam: %.4f Best R: %.4f Best Thresh: %.4f" % (b_lam, b_R, b_thresh)))
    run_trial(utils, Input, x, b_thresh, b_R, b_lam, plot=True)

# run a trial and return the l2 norm as a cost
def run_trial(utils, Input, x, delta_mod_tresh, R, lam, plot=False):
    (OT_down, OT_up) = get_spiking_input(delta_mod_tresh, Input, utils.Nx, utils.Ntime)
    x_recon = np.zeros(Input.shape)
    for t in range(1,utils.Ntime):
         x_recon[:,t] = (1-lam)*x_recon[:,t-1] + R*np.asarray([OT_up[0,t]-OT_down[0,t], OT_up[1,t]-OT_down[1,t]])    

    if(plot):
        plt.plot(x.T)
        plt.plot(x_recon.T)
        plt.show()
    return np.linalg.norm((x-x_recon).reshape(-1,),2)



def Learning(utils, F, C, conn_x_down, conn_x_high):

    TotTime = utils.Nit*utils.Ntime

    spiking_input_v_cont_old = np.zeros((utils.Nneuron,))

    Fi = np.copy(F)
    Ci = np.copy(C)

    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Fs = np.zeros([utils.T, utils.Nx, utils.Nneuron]) # Store the FF weights over the course of training

    V = np.zeros((utils.Nneuron, 1))

    O = 0
    k = 0 #! Indexing starts with 0
    r0 = np.zeros((utils.Nneuron, 1))

    x = np.zeros((utils.Nx, 1))
    xs = np.zeros((utils.Nx, utils.Ntime))
    xs_recon = np.zeros((utils.Nx, utils.Ntime))
    x_recon = np.zeros((utils.Nx, 1))
    Input = np.zeros((utils.Nx, utils.Ntime))
    Id = np.eye(utils.Nneuron)

    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)

    j = 1; l = 1

    print(("%d percent of learning done" % 0))

    for i in range(2, TotTime):

        if((i/TotTime) > (l/100)):
            print(("%d percent of learning done" % l))
            l = l+1

        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = C # Indexing starts at 0
            Fs[j-1,:,:] = F
            j = j+1

        if(((i-2) % utils.Ntime) == 0):
            Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
            for d in range(utils.Nx):
                Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

            (OT_down, OT_up) = get_spiking_input(utils.delta_modulator_threshold, Input, utils.Nx, utils.Ntime)



        #! julianb, spiking input
        """I = (1-lam*dt)*I + dt*F_spikes^T*Input_spikes + O*C[:,k] + 0.001*randn(NNeuron)
        I = (1-utils.R*utils.dt)*I + utils.dt*np.matmul(F_spikes.T, Input_spikes[:,(i % utils.Ntime)].reshape((-1,1))) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
        V = (1-utils.dt)*V + utils.dt*utils.R*I
        V[V >= utils.Thresh] = V[V >= utils.Thresh] - utils.Thresh*np.ones(len(V[V >= utils.Thresh]))"""
        
        t = (i % utils.Ntime)

        x = (1-utils.lam*utils.dt)*x + utils.dt*Input[:, t].reshape((-1,1)) #! Removed (i % Ntime)+1 the +1 for indexing
        xs[:,t] = x.ravel()

        # conn_x_high[0] is the first F for the first up_spike_train
        spiking_input_v_cont = spiking_input_v_cont_old + conn_x_high[0]*OT_up[0,t] + conn_x_high[1]*OT_up[1,t]-conn_x_down[0]*OT_down[0,t]-conn_x_down[1]*OT_down[1,t]

        lam = 0.001; R = 1.0 
        x_recon = (1-lam)*x_recon + R*np.asarray([OT_up[0,t]-OT_down[0,t], OT_up[1,t]-OT_down[1,t]]).reshape((-1,1))    
        xs_recon[:,t] = x_recon.ravel()

        #V = (1-utils.lam*utils.dt)*V + utils.dt*np.matmul(F.T, Input[:,(i % utils.Ntime)].reshape((-1,1))) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
        #V = (1-utils.lam*utils.dt)*V + spiking_input_v_cont.reshape((-1,1)) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
        V = (1-utils.lam*utils.dt)*V + 0.1*np.matmul(F.T, x_recon.reshape((-1,1))) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
        
        spiking_input_v_cont_old = spiking_input_v_cont

        (m, k) = my_max(V - utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)) # Returns maximum and argmax

        if (m >= 0): # We have a spike
            O = 1
            # F[:,k] = (F[:,k].reshape((-1,1)) + utils.epsf*(utils.alpha*x - F[:,k].reshape((-1,1)))).ravel()
            C[:,k] = (C[:,k].reshape((-1,1)) - utils.epsr*(utils.beta*(V + utils.mu*r0) + C[:,k].reshape((-1,1)) + utils.mu*Id[:,k].reshape((-1,1)))).ravel()
            r0[k] = r0[k] + 1
        else:
            O = 0
        
        r0 = (1-utils.lam*utils.dt)*r0


    print("Learning complete")

        
    ########## Compute the optimal decoder ##########

    TimeL = 50000
    xL = np.zeros((utils.Nx, TimeL))
    Decs = np.zeros([utils.T, utils.Nx, utils.Nneuron])

    # Generate new input
    InputL = 0.3*utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeL)).T
    for d in range(utils.Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')
        
    #! Get the long spiking input for the decoders    
    (OT_downL, OT_upL) = get_spiking_input(utils.delta_modulator_threshold, InputL, utils.Nx, TimeL)

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    print(("Computing %d decoders" % utils.T))

    for i in range(utils.T):
        #! Use spiking input for decoder
        #(rOL,_,_) =  runnet_spike_input(utils.dt, utils.lam, conn_x_high, conn_x_down, OT_upL, OT_downL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh)
        #(rOL,_,_) = runnet(utils.dt, utils.lam, Fs[i,:,:], InputL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh)
        (rOL,_,_) = runnet_recon_x(utils.dt, utils.lam, Fs[i,:,:], OT_upL, OT_downL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh, x_recon_lam = 0.001, x_recon_R = 1.0)
        Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        Decs[i,:,:] = Dec

    print("Computing the errors")
    TimeT = 10000
    MeanPrate = np.zeros((1,utils.T))
    Error = np.zeros((1,utils.T))
    MembraneVar = np.zeros((1,utils.T))
    xT = np.zeros((utils.Nx, TimeT))

    Trials = 5

    for r in range(Trials):
        InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T

        for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

        (OT_downT, OT_upT) = get_spiking_input(utils.delta_modulator_threshold, InputT, utils.Nx, TimeT)

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

        for i in range(utils.T):
            #(rOT, OT, VT) = runnet(utils.dt, utils.lam, Fs[i,:,:], InputT, Cs[i,:,:], utils.Nneuron, TimeT, utils.Thresh)
            #(rOT, OT, VT) = runnet_spike_input(utils.dt, utils.lam, conn_x_high, conn_x_down, OT_upT, OT_downT, Cs[i,:,:], utils.Nneuron, TimeT, utils.Thresh)
            (rOT, OT, VT) = runnet_recon_x(utils.dt, utils.lam, Fs[i,:,:], OT_upT, OT_downT, Cs[i,:,:], utils.Nneuron, TimeT, utils.Thresh, x_recon_lam = 0.001, x_recon_R = 1.0)
            xestc = np.matmul(Decs[i,:,:], rOT) # Decode the rate vector
            Error[0,i] = Error[0,i] + np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)
            MeanPrate[0,i] = MeanPrate[0,i] + np.sum(OT) / (TimeT*utils.dt*utils.Nneuron*Trials)
            MembraneVar[0,i] = MembraneVar[0,i] + np.sum(np.var(VT, axis=1, ddof=1)) / (utils.Nneuron*Trials)


    ErrorC = np.zeros((1,utils.T))
    for i in range(utils.T):
        CurrF = Fs[i,:,:]
        CurrC = Cs[i,:,:]

        Copt = np.matmul(-CurrF.T, CurrF)
        optscale = np.trace(np.matmul(CurrC.T, Copt)) / np.sum(Copt**2)
        Cnorm = np.sum(CurrC**2)
        ErrorC[0,i] = np.sum(np.sum((CurrC - optscale*Copt)**2 ,axis=0)) / Cnorm
    
    ########## Plotting ##########

    return_dict = {
        "Fi": Fi,
        "Ci": Ci,
        "Deci": Decs[0,:,:],
        "F_after": F,
        "C_after": C,
        "D_after": Dec,
        "Cs": Cs,
        "Fs": Fs,
        "Decs": Decs,
        "Error": Error,
        "MeanPrate": MeanPrate,
        "MembraneVar": MembraneVar,
        "ErrorC": ErrorC,
        "w": w
        }

    return return_dict



