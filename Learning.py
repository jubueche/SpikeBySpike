import numpy as np
import matplotlib.pyplot as plt
from Utils import my_max, ups_downs_to_O
from runnet import *
from progress.bar import ChargingBar
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DYNAPS/"))
from helper import signal_to_spike_refractory


def discretize(C, min, max, number_of_bins):
    """
    Parameters:
                C:              Matrix of shape (Nneurons,Nneurons)
                min:            The minimum bin. All weights in or below that bin border are assigned the most negative,
                                discrete weight
                max:            The maximum bin
                number_of_bins: Number of bins. The higher the number, the higher the precision.
    Returns:
                Discretized version of C
    """
    _, bin_edges = np.histogram(C.reshape((-1,1)), bins = number_of_bins, range=(min,max))
    bin_indices = np.digitize(C.ravel(), bins = bin_edges, right = True)
    C_new_discretized = np.zeros(C.shape[0]**2)
    for i in range(C.shape[0]**2):
        if(bin_indices[i] >= len(bin_edges)):
            C_new_discretized[i] = bin_edges[bin_indices[i]-1]
        else:
            C_new_discretized[i] = bin_edges[bin_indices[i]]
    
    C_new_discretized = C_new_discretized.reshape(C.shape)
    np.fill_diagonal(C_new_discretized, np.diagonal(C))
    return C_new_discretized


def Learning(utils, F, C):
    """
    Parameters:
                utils:   Object that holds various parameters used for training
                F:       Initial feedforward matrix (real)
                C:       Initial recurrent matrix (real)
    Returns:  
                results: Dictionary containing all trained and relevant variables
    """
    TotTime = utils.Nit*utils.Ntime

    Fi = np.copy(F)
    Ci = np.copy(C)

    # Parameters for discretization
    min = -0.35
    max = 0.42
    # number_of_bins = 100

    # C_initial_discretized = discretize(np.copy(Ci), min = min, max = max, number_of_bins= number_of_bins)
    # C = discretize(C, min, max, number_of_bins)

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
    OS = np.zeros((utils.Nneuron, utils.Ntime))
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

    j = 1

    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

    Input.dump("DYNAPS/Resources/bias_input.dat")

    bar = ChargingBar('Learning', max=TotTime-1)
    for i in range(2, TotTime):

        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = C # Indexing starts at 0
            Fs[j-1,:,:] = F
            j = j+1

        if(((i-2) % utils.Ntime) == 0):
            # Generate new input
            Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
            for d in range(utils.Nx):
                Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

            if(i == utils.Ntime + 2):
                OS.dump("DYNAPS/Resources/target_OS.dat")

            (OT_down, OT_up) = get_spiking_input(utils.delta_modulator_threshold, Input, utils.Nx, utils.Ntime)
        
            # Need to do the update
            # First transform each column
            for i in range(delta_Omega.shape[1]):
                if(ks[i] > 0):
                    delta_Omega[:,i] /= ks[i]
            # Do the update
            C = C - delta_Omega
            # C = discretize(C, min, max, number_of_bins)
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

        # V = (1-utils.lam*utils.dt)*V + delta_F*FTMI.reshape((-1,1)) + O*C[:,k].reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)        
        # Vs[:,t] = V.ravel()
        V_recons[:,t] = V_recon.ravel()

        current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)

        # Update the reconstructed voltage using V_recon = F.T*x(t) + Omega*r(t-1)
        new_V_recon = 0.1*V_recon + np.matmul(F.T, x) + np.matmul(C, r0)
        # new_V_recon = 0.1*V_recon + FTMI + np.matmul(C, r0)

        (m, k) = my_max(new_V_recon - current_thresh) # Returns maximum and argmax

        diff = (new_V_recon - current_thresh)
        neurons_above = np.linspace(0,utils.Nneuron-1, utils.Nneuron)[diff.ravel() >= 0].astype(np.int)

        if (m >= 0): # We have a spike
            O = 1
            # F[:,k] = (F[:,k].reshape((-1,1)) + utils.epsf*(utils.alpha*x - F[:,k].reshape((-1,1)))).ravel()
            #tmp = (utils.epsr*(utils.beta*(V + utils.mu*r0) + C[:,k].reshape((-1,1)) + utils.mu*Id[:,k].reshape((-1,1)))).ravel()
            # Update using the reconstructed voltage V_recon(t-1)
            tmp = (utils.epsr*(utils.beta*(new_V_recon + utils.mu*r0) + C[:,k].reshape((-1,1)) + utils.mu*Id[:,k].reshape((-1,1)))).ravel()
            delta_Omega[:,k] =  delta_Omega[:,k] + tmp
            ks[k] += 1
            r0[k] = r0[k] + 1
            OS[k,t] = 1
        else:
            O = 0
            
        r0 = (1-utils.lam*utils.dt)*r0
        
        Rs[:,t] = r0.ravel()

        # Assign the new reconstructed voltage
        V_recon = new_V_recon

        bar.next()
    bar.next()
    bar.finish()

        
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

    print("")
    bar = ChargingBar('Decoders', max=utils.T)
    for i in range(utils.T):
        (rOL,_,_) = runnet_recon_x(utils.dt, utils.lam, Fs[i,:,:], OT_upL, OT_downL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh, xL, x_recon_lam = x_recon_lam, x_recon_R = x_recon_R, delta_F=delta_F)
        Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        Decs[i,:,:] = Dec
        bar.next()
    bar.next()
    bar.finish()
    print("")

    TimeT = 1000 #! Was 10000
    MeanPrate = np.zeros((1,utils.T))
    Error = np.zeros((1,utils.T))
    MembraneVar = np.zeros((1,utils.T))
    xT = np.zeros((utils.Nx, TimeT))

    Trials = 5
    bar = ChargingBar('Errors', max=Trials)
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
        bar.next()
    bar.next()
    bar.finish()

    ErrorC = np.zeros((1,utils.T))
    for i in range(utils.T):
        CurrF = Fs[i,:,:]
        CurrC = Cs[i,:,:]

        Copt = np.matmul(-CurrF.T, CurrF)
        optscale = np.trace(np.matmul(CurrC.T, Copt)) / np.sum(Copt**2)
        Cnorm = np.sum(CurrC**2)
        ErrorC[0,i] = np.sum(np.sum((CurrC - optscale*Copt)**2 ,axis=0)) / Cnorm
    
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



