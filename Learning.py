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
    threshs = np.linspace(0.5, 80, 100)
    old_err = -1
    for lam in lams:
        for R in Rs:
            for thresh in threshs:
                # err = run_trial(utils, Input, x, thresh, R, lam)
                err = run_trial_FTMI(utils, Input, x, thresh, R, lam)
                if(old_err == -1 or err < old_err):
                    print(err)
                    old_err = err
                    b_lam = lam; b_R = R; b_thresh = thresh

    print(("Best lam: %.4f Best R: %.4f Best Thresh: %.4f" % (b_lam, b_R, b_thresh)))
    run_trial_FTMI(utils, Input, x, b_thresh, b_R, b_lam, plot=True)


def run_trial_FTMI(utils, Input, x, delta_mod_tresh, R, lam, plot=False):
    M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    FTMI = np.zeros((utils.Nneuron, 1))
    I = np.zeros((2*utils.Nx, utils.Ntime))
    (OT_down, OT_up) = get_spiking_input(delta_mod_tresh, Input, utils.Nx, utils.Ntime)
    for t in range(1, utils.Ntime):
        ot = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
        I[:,t] = ((1-lam)*I[:,t-1].reshape((-1,1)) + R*ot).ravel()

    x_recon = np.matmul(M, I)
    if(plot):
        plt.plot(x.T)
        plt.plot(x_recon.T)
        plt.title("Signal reconstructed from spiking input")
        plt.show()
    return np.linalg.norm((x-x_recon).reshape(-1,),2)



def Learning(utils, F, C):

    TotTime = utils.Nit*utils.Ntime

    Fi = np.copy(F)
    Ci = np.copy(C)

    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Fs = np.zeros([utils.T, utils.Nx, utils.Nneuron]) # Store the FF weights over the course of training

    M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    MIs = np.zeros((utils.Nx, utils.Ntime))
    x_recon_lam = 0.001
    x_recon_R = 1.0
    delta_F = 0.1
    # Save the updates in here
    delta_Omega = np.zeros(C.shape)
    # Keep track of how many times C[:,k] was updated
    ks = np.zeros(C.shape[1])
    # Reconstructed voltage using v_r(t) = F.T*x(t) + Omega*r(t-1)
    Vs = np.zeros((utils.Nneuron, utils.Ntime))
    V_recons = np.zeros((utils.Nneuron, utils.Ntime))
    Rs = np.zeros((utils.Nneuron, utils.Ntime))
    V_recon = np.zeros((utils.Nneuron, 1))
    new_V_recon = np.zeros((utils.Nneuron, 1)) # Temporary storage for the new V_recon
    # Store the current threshold
    current_thresh = np.zeros((utils.Nneuron, 1))

    V = np.zeros((utils.Nneuron, 1))

    O = 0
    k = 0 #! Indexing starts with 0
    r0 = np.zeros((utils.Nneuron, 1))

    x = np.zeros((utils.Nx, 1))
    I = np.zeros((2*utils.Nx, 1))
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
        
            # Need to do the update
            # First transform each column
            for i in range(delta_Omega.shape[1]):
                if(ks[i] > 0):
                    delta_Omega[:,i] /= ks[i]
            # Do the update
            C = C - delta_Omega
            # Reset
            ks = np.zeros(delta_Omega.shape[1])
            delta_Omega = np.zeros(C.shape)

        t = (i % utils.Ntime)

        # Filter the signal to get smaller scale
        x = (1-utils.lam*utils.dt)*x + utils.dt*Input[:, t].reshape((-1,1))

        ot = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
        I = (1-x_recon_lam)*I + x_recon_R*ot
        MIs[:,t] = np.matmul(M, I).ravel()
        FTMI = np.matmul(np.matmul(F.T, M), I)

        V = (1-utils.lam*utils.dt)*V + delta_F*FTMI.reshape((-1,1)) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)        
        Vs[:,t] = V.ravel()
        V_recons[:,t] = V_recon.ravel()

        current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)

        # Update the reconstructed voltage using V_recon = F.T*x(t) + Omega*r(t-1)
        new_V_recon = 0.1*V_recon + np.matmul(F.T, x) + np.matmul(C, r0)
        # new_V_recon = 0.1*V_recon + FTMI + np.matmul(C, r0)
        (m, k) = my_max(new_V_recon - current_thresh) # Returns maximum and argmax

        # Do the integration reset of the reconstructed voltage
        for i in range(utils.Nneuron):
            diff = V_recon[i] - new_V_recon[i]
            if(diff > utils.Thresh - 0.05):
                V_recon[i] = current_thresh[i] - diff

        if (m >= 0): # We have a spike
            O = 1
            # F[:,k] = (F[:,k].reshape((-1,1)) + utils.epsf*(utils.alpha*x - F[:,k].reshape((-1,1)))).ravel()
            #tmp = (utils.epsr*(utils.beta*(V + utils.mu*r0) + C[:,k].reshape((-1,1)) + utils.mu*Id[:,k].reshape((-1,1)))).ravel()
            # Update using the reconstructed voltage V_recon(t-1)
            tmp = (utils.epsr*(utils.beta*(new_V_recon + utils.mu*r0) + C[:,k].reshape((-1,1)) + utils.mu*Id[:,k].reshape((-1,1)))).ravel()
            delta_Omega[:,k] =  delta_Omega[:,k] + tmp
            ks[k] += 1
            r0[k] = r0[k] + 1
        else:
            O = 0
        
        r0 = (1-utils.lam*utils.dt)*r0
        Rs[:,t] = r0.ravel()

        # Assign the new reconstructed voltage
        V_recon = np.copy(new_V_recon) #! Try w/o copy


    print("Learning complete")

        
    ########## Compute the optimal decoder ##########

    TimeL = 5000 #50000 #! Was 50k
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
        (rOL,_,_) = runnet_recon_x(utils.dt, utils.lam, Fs[i,:,:], OT_upL, OT_downL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh, xL, x_recon_lam = x_recon_lam, x_recon_R = x_recon_R, delta_F=delta_F)
        Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        Decs[i,:,:] = Dec

    print("Computing the errors")
    TimeT = 1000 #! Was 10000
    MeanPrate = np.zeros((1,utils.T))
    Error = np.zeros((1,utils.T))
    MembraneVar = np.zeros((1,utils.T))
    xT = np.zeros((utils.Nx, TimeT))

    Trials = 2

    for r in range(Trials):
        InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T

        for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

        (OT_downT, OT_upT) = get_spiking_input(utils.delta_modulator_threshold, InputT, utils.Nx, TimeT)

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

        for i in range(utils.T):
            (rOT, OT, VT) = runnet_recon_x(utils.dt, utils.lam, Fs[i,:,:], OT_upT, OT_downT, Cs[i,:,:], utils.Nneuron, TimeT, utils.Thresh, xT, x_recon_lam = x_recon_lam, x_recon_R = x_recon_R, delta_F=delta_F)
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



