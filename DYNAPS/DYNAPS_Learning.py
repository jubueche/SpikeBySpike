import numpy as np
import matplotlib.pyplot as plt
from DYNAPS_runnet import runnet, my_max
from progress.bar import ChargingBar
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"../"))
from runnet import get_spiking_input

def align_slow_signal(sbs, utils):
    """
    This function produces a very slow changing signal and computes the reconstructed voltages from it using a filtered
    version of the spike inputs. If the reconstructed voltages cross a threshold, a spike is triggered.
    Unlike in the simulation, we trigger a spike for every neuron that crosses the threshold.
    The next step would be to find parameters on the DYNAPS that reproduce the spike train 100% and yield a very small
    difference in the rate vectors.
    """

    np.random.seed(42)

    w = (1/(100*utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*(100*utils.sigma)**2))
    w = w / np.sum(w)
    # Calculate and load the signal for the decoders
    TimeL = 1000 # Was 50000 in the simulation

    Nx = 1

    xL = np.zeros((Nx, TimeL))

    InputL = utils.A*(np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)).T
    for d in range(Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    (OT_down, OT_up) = get_spiking_input(0.1, xL, Nx, TimeL)
    
    """if(sbs.debug): 
        coordinates_up = np.nonzero(OT_up)
        coordinates_dwn = np.nonzero(OT_down)
        plt.figure(figsize=(6,4))
        plt.subplot(311)
        plt.scatter(coordinates_up[1], coordinates_up[0], marker='o', s=0.5)
        plt.xlim((0,TimeL))
        plt.subplot(312)
        plt.scatter(coordinates_dwn[1], coordinates_dwn[0], marker='o', s=0.5)
        plt.xlim((0,TimeL))
        plt.subplot(313)
        plt.plot(xL.T)
        plt.xlim((0,TimeL))
        plt.tight_layout()
        plt.show()"""


    Is = np.zeros((2*Nx, 1))
    Vs = np.zeros((utils.Nneuron,TimeL))
    OT = np.zeros((utils.Nneuron,TimeL))
    I = np.zeros(Nx)
    M = np.asarray([1,-1]).reshape((1,-1))
    MIs = np.zeros((Nx, TimeL))

    F = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    F = utils.gamma*np.divide(F, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F**2, axis=0).reshape((1,utils.Nneuron)))))
    F = F[0,:].reshape((1,-1))
    C = -np.eye(utils.Nneuron)*0.5

    ot = np.zeros(utils.Nneuron)
    k = 0

    FtM = np.matmul(F.T, M)
    for i in range(FtM.shape[1]): # for all columns
        FtM[:,i] = FtM[:,i] / (max(FtM[:,i]) - min(FtM[:,i])) * 2*utils.dynapse_maximal_synapse_ff
    FtM = np.asarray(FtM, dtype=int)

    for t in range(1,TimeL):

        current_threshold = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)

        ot = np.asarray([OT_up[0,t], OT_down[0,t]])
        I = (1-0.000001)*I + 0.1*ot
        MIs[:,t] = np.matmul(M, I).ravel()
        FtMI = np.matmul(np.matmul(F.T, M), I).reshape((-1,))
        ot = OT[:,t-1].reshape((-1,1))

        #Vs[:,t] = (1-utils.lam*utils.dt)*Vs[:,t-1] + np.matmul(F.T, xL[:,t]).ravel() + np.matmul(C,ot).ravel() + 0.001*np.random.randn(utils.Nneuron, 1).ravel()
        Vs[:,t] = (1-utils.lam*utils.dt)*Vs[:,t-1] + FtMI + np.matmul(C,ot).ravel() + 0.001*np.random.randn(utils.Nneuron, 1).ravel()

        diff = (Vs[:,t] - current_threshold.ravel())
        (m,k) = my_max(diff)
        if(m>=0):
            OT[k,t] = 1

        """neurons_that_spiked = (diff >= 0)
        OT[neurons_that_spiked, t] = 1"""

    """sbs.F = FtM
    sbs.set_recurrent_connection()"""

    """plt.plot(MIs.T)
    plt.plot(xL.T)
    plt.show()"""

    coordinates = np.nonzero(OT)
    plt.scatter(coordinates[1], coordinates[0], marker='o', s=0.5)
    plt.show()

def tune_biases(sbs, utils):

    F = np.load("Resources/F_initial.dat", allow_pickle=True)
    F_disc = np.load("Resources/DYNAPS_F.dat", allow_pickle=True)
    C_inital = np.load("Resources/Ci.dat", allow_pickle=True)
    C_after = np.load("Resources/C_after.dat", allow_pickle=True)

    # 1) Set F_disc
    sbs.F = np.copy(F_disc)
    max_C = 0.625
    min_C = -0.545

    # 2)
    # Kernel
    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)
    # Calculate and load the signal for the decoders
    TimeL = 5000 # Was 50000 in the simulation
    xL = np.zeros((utils.Nx, TimeL))

    InputL = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeL)).T
    for d in range(utils.Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    if(sbs.debug):
        plt.plot(xL.T)
        plt.show()

    num_parameters = 5
    fine_inh = [200,20,100,50,255]
    coarse_inh = [5,6,4,-1,-1]
    delta_mod_ups = [0.01,0.001,0.005,0.05,-1]
    delta_mod_dwns = [0.001,0.01,0.005,0.05,-1]
    best = 0.0

    best_fi = fine_inh[0]
    best_ci = coarse_inh[0]
    best_dup = delta_mod_ups[0]
    best_ddwn = delta_mod_dwns[0]

    order = ['ci', 'fi', 'dup', 'ddwn']
    T = 4
    for t in range(T):
        to_tune = order[t % len(order)]
        print("Tuning %s" % to_tune)
        for i in range(num_parameters):

            if (to_tune == 'fi'):
                ci = best_ci; dup = best_dup; ddwn = best_ddwn; fi = fine_inh[i]
                if (fi == -1): break
            elif (to_tune == 'ci'):
                ci = coarse_inh[i]; dup = best_dup; ddwn = best_ddwn; fi = best_fi
                if (ci == -1): break
            elif (to_tune == 'dup'):
                ci = best_ci; dup = delta_mod_ups[i]; ddwn = best_ddwn; fi = best_fi
                if (dup == -1): break
            elif (to_tune == 'ddwn'):
                ci = best_ci; dup = best_dup; ddwn = delta_mod_dwns[i]; fi = best_fi
                if(ddwn == -1): break
            else:
                raise Exception("No element to fine tune")

            sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
            sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", fi, ci) # 200, 5
            delta_mod_thresh_up = dup
            delta_mod_thresh_dwn = ddwn

            # Load the new input into the spike generator
            sbs.load_signal(xL, delta_mod_thresh_up, delta_mod_thresh_dwn)

            # 3)
            C_i_discrete = sbs.bin_omega(C_inital, min = min_C, max = max_C)
            (rOL,_,_) = runnet(sbs, utils, F, C_i_discrete, C_inital, TimeL, xL)
            D_initial = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
            C_after_discrete = sbs.bin_omega(C_after, min = min_C, max = max_C)
            (rOL,_,_) = runnet(sbs, utils, F, C_after_discrete, C_after, TimeL, xL)
            D_after = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T

            # 4)
            trials = 5
            error_diff = np.zeros(5)
            TimeT = 1000
            xT = np.zeros((utils.Nx, TimeT))
            for i in range(trials):
                # Generate new signal to test on
                InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T
                for d in range(utils.Nx):
                    InputT[d,:] = np.convolve(InputT[d,:], w, 'same')
                # Compute the target output by leaky integration of InputT
                for t in range(1,TimeT):
                    xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]
                # Load the input
                sbs.load_signal(np.copy(xT), delta_mod_thresh_up, delta_mod_thresh_dwn)
                (rOT, OT, VT) = runnet(sbs, utils, F, C_i_discrete, C_inital, TimeT, xT)
                xestc = np.matmul(D_initial, rOT) # Decode the rate vector
                err_initial = np.sum(np.var(xT-xestc, axis=1, ddof=1)) / np.sum(np.var(xT, axis=1, ddof=1))
                (rOT, OT, VT) = runnet(sbs, utils, F, C_after_discrete, C_after, TimeT, xT)
                xestc = np.matmul(D_after, rOT) # Decode the rate vector
                err_after = np.sum(np.var(xT-xestc, axis=1, ddof=1)) / np.sum(np.var(xT, axis=1, ddof=1))
                error_diff[i] = err_after - err_initial

            if(sbs.debug):
                print(error_diff)
                print(np.mean(error_diff))
            
            if(np.mean(error_diff) < best):
                if (to_tune == 'fi'):
                    best_fi = fi
                elif (to_tune == 'ci'):
                    best_ci = ci
                elif (to_tune == 'dup'):
                    best_dup = dup
                elif (to_tune == 'ddwn'):
                    best_ddwn = ddwn
                else:
                    raise Exception("No element to fine tune")
                best = np.mean(error_diff)
                print("Fi: %d Ci: %d Dup: %.4f Ddwn: %.4f" % (best_fi,best_ci,best_dup,best_ddwn))
    



def Learning(sbs, utils, F, FtM, C, debug = False):
    print("Setting FF...")
    sbs.F = np.copy(FtM)
    max_C = 0.625
    min_C = -0.545
    delta_t = 20
    
    # Setting the weights on DYNAPS
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 255, 5) # 200, 5

    # Total training time
    TotTime = utils.Nit*utils.Ntime
    # Copy the initial recurrent weights
    Ci = np.copy(C)
    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Cs_discrete = np.zeros([utils.T, utils.Nneuron, utils.Nneuron])
    C_current_discrete = sbs.bin_omega(C_real=np.copy(C), min=min_C, max=max_C)

    delta_mod_thresh_up = 0.05 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.05
    # Save the updates in here
    delta_Omega = np.zeros(C.shape)
    # Keep track of how many times C[:,k] was updated
    ks = np.zeros(C.shape[1])

    V_recons = np.zeros((utils.Nneuron, utils.Ntime))
    new_V_recon = np.zeros((utils.Nneuron, 1)) # Temporary storage for the new V_recon
    # Store the current threshold
    current_thresh = np.zeros((utils.Nneuron, 1))

    k = 0

    # Computed rates over course of one signal
    R = np.zeros((utils.Nneuron, utils.Ntime))
    # Population spike train of DYNAPS
    O_DYNAPS = np.zeros((utils.Nneuron, utils.Ntime))
    X = np.zeros((utils.Nx, utils.Ntime))

    Input = np.zeros((utils.Nx, utils.Ntime))
    Id = np.eye(utils.Nneuron)

    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)

    j = 1

    bar = ChargingBar('Learning', max=TotTime-1)
    for i in range(2, TotTime):

        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = C # Indexing starts at 0
            Cs_discrete[j-1,:,:] = C_current_discrete
            j = j+1

        """
        1) Get new input
        2) Filter the input and save in X
        3) Load the input into the spike generator
        4) Update Omega based on previously calculated Delta Omega
        5) Set Omega weight in sbs controller
        6) Run Trial and save population spike train in O_DYNAPS
        7) Compute the rate vectors from O_DYNAPS and store in R
        8) Compute the reconstructed voltages using lam*V(t-1) + F.T*x(t) + C*r(t-1)
           and accumulate delta Omega.
        """
        if(((i-2) % utils.Ntime) == 0):
            Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
            for d in range(utils.Nx):
                Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

            # 2)
            X = np.zeros(X.shape)
            for t in range(1, utils.Ntime):
                X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:, t]
            """if(sbs.debug):
                plt.plot(X.T)
                plt.show()"""
            # 3)
            sbs.load_signal(np.copy(X), delta_mod_thresh_up, delta_mod_thresh_dwn)
            # 4)
            for i in range(delta_Omega.shape[1]): # First transform each column
                if(ks[i] > 0):
                    delta_Omega[:,i] /= ks[i]
            # Do the update
            C = C - delta_Omega
            
            C_current_discrete = sbs.bin_omega(C_real=np.copy(C), min=min_C, max=max_C)
            if(sbs.debug):
                print(C_current_discrete)
            # 5)
            sbs.set_recurrent_connection()

            # Reset
            ks = np.zeros(delta_Omega.shape[1])
            delta_Omega = np.zeros(C.shape)
            # 6)
            O_DYNAPS = sbs.execute()
            """# 7) #! In the simulation, we only update the rate vec at one point per time step
            for t in range(1, utils.Ntime):
                R[:,t] = (1-utils.lam*utils.dt)*R[:,t-1] + O_DYNAPS[:,t]
            # 8)
            for t in range(1, utils.Ntime):
                current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
                #! new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, X[:,t]) + np.matmul(C, R[:,t-1])
                new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, X[:,t]) + R[:,t] # Rates already incorporate the recurrent connections

                (m, k) = my_max(new_V_recon.reshape((-1,1)) - current_thresh) # Returns maximum and argmax
                if(m[0] >= 0):
                    tmp = utils.epsr*(utils.beta*(new_V_recon + utils.mu*R[:,t-1]) + C[:,k] + utils.mu*Id[:,k])
                    delta_Omega[:,k] += tmp
                    ks[k] += 1
                V_recons[:,t] = new_V_recon"""

            for t in range(1, utils.Ntime):
                current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
                new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, X[:,t]) + np.matmul(C, R[:,t-1])

                (m, k) = my_max(new_V_recon.reshape((-1,1)) - current_thresh) # Returns maximum and argmax
                has_spike = np.sum(O_DYNAPS[k,max(0,t-delta_t):min(utils.Ntime-1,t+delta_t)]) > 0
                r_tmp = R[:,t]
                if(m[0] >= 0 and has_spike):
                    tmp = utils.epsr*(utils.beta*(new_V_recon + utils.mu*R[:,t-1]) + C[:,k] + utils.mu*Id[:,k])
                    delta_Omega[:,k] += tmp
                    ks[k] += 1
                    r_tmp[k] += 1

                V_recons[:,t] = new_V_recon
                R[:,t] = (1-utils.lam*utils.dt)*r_tmp

            """if(sbs.debug):
                variance = np.sum(np.var(V_recons, axis=1, ddof=1)) / (utils.Nneuron)
                print(variance)
                plt.plot(V_recons[5,:])
                plt.show()
                plt.plot(R[5,:])
                plt.show()"""
            
        bar.next()
    bar.next()
    bar.finish()

    ########## Compute the optimal decoder ##########
    TimeL = 5000 # Was 50000 in the simulation
    xL = np.zeros((utils.Nx, TimeL))
    Decs = np.zeros([utils.T, utils.Nx, utils.Nneuron])

    # Generate new input
    #InputL = 0.3*utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeL)).T
    InputL = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeL)).T
    for d in range(utils.Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    # Load the new input into the spike generator
    print("Loading new signal...")
    sbs.load_signal(xL, delta_mod_thresh_up, delta_mod_thresh_dwn)

    print(("Computing %d decoders" % utils.T))

    bar = ChargingBar('Decoders', max=utils.T)

    for i in range(utils.T):
        if(np.sum(Cs[i,:,:]- Ci) == 0):
            if(i > 0): # Copy from the previous
                Dec = Decs[i-1,:,:]
            else:
                (rOL,_,_) = runnet(sbs, utils, F, Cs_discrete[i,:,:], Cs[i,:,:], TimeL, xL)
                Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        else:
            (rOL,_,_) = runnet(sbs, utils, F, Cs_discrete[i,:,:], Cs[i,:,:], TimeL, xL)
            Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        Decs[i,:,:] = Dec
        bar.next()
    bar.finish()

    print("Computing the errors")
    TimeT = 1000 #! Was 10000
    MeanPrate = np.zeros((1,utils.T))
    Error = np.zeros((1,utils.T))
    MembraneVar = np.zeros((1,utils.T))
    xT = np.zeros((utils.Nx, TimeT))

    Trials = 5
    for r in range(Trials):
        InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T

        for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]
        
        # Load the input
        sbs.load_signal(np.copy(xT), delta_mod_thresh_up, delta_mod_thresh_dwn)

        bar = ChargingBar(('Error #%d' % r), max=utils.T)
        for i in range(utils.T):
            if(sbs.debug):
                print(Cs_discrete[i,:,:])
            print(Cs_discrete[i,:,:])
            print(np.sum(np.abs(Cs_discrete[i,:,:])))
            (rOT, OT, VT) = runnet(sbs, utils, F, Cs_discrete[i,:,:], Cs[i,:,:], TimeT, xT)
            xestc = np.matmul(Decs[i,:,:], rOT) # Decode the rate vector
            
            err = np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)
            print("Error is %d", err)           
            Error[0,i] = Error[0,i] + err
            MeanPrate[0,i] = MeanPrate[0,i] + np.sum(OT) / (TimeT*utils.dt*utils.Nneuron*Trials)
            MembraneVar[0,i] = MembraneVar[0,i] + np.sum(np.var(VT, axis=1, ddof=1)) / (utils.Nneuron*Trials)
            bar.next()
        bar.finish()
        print("")

    ErrorC = np.zeros((1,utils.T))
    for i in range(utils.T):
        CurrC = Cs[i,:,:]

        Copt = np.matmul(-F.T, F)
        optscale = np.trace(np.matmul(CurrC.T, Copt)) / np.sum(Copt**2)
        Cnorm = np.sum(CurrC**2)
        ErrorC[0,i] = np.sum(np.sum((CurrC - optscale*Copt)**2 ,axis=0)) / Cnorm

    return_dict = {
        "Fi": F,
        "Ci": Ci,
        "Deci": Decs[0,:,:],
        "F_after": F,
        "C_after": C,
        "D_after": Dec,
        "Cs": Cs,
        "Decs": Decs,
        "Error": Error,
        "MeanPrate": MeanPrate,
        "MembraneVar": MembraneVar,
        "ErrorC": ErrorC,
        "w": w
        }

    return return_dict