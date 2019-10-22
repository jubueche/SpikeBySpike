import numpy as np
import matplotlib.pyplot as plt
from DYNAPS_runnet import runnet, my_max
from progress.bar import ChargingBar

def find_biases(sbs, utils, F_disc, C):

    max_C = 0.412
    min_C = -0.339
    
    # Bias group index: 4
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 107, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 107, 7)

    FtM_sim = np.load("Resources/DYNAPS_F.dat", allow_pickle=True)
    assert np.sum((FtM_sim - F_disc)) == 0, "Assertion failed. Matrices not equal"

    # Load the FF matrix
    sbs.F = np.copy(F_disc)

    # We will test the behaviour on a test input that was computed by the simulation
    Input = np.load("Resources/bias_input.dat", allow_pickle=True)
    # Load the target rate matrix that we want to reconstruct
    target_O = np.load("Resources/target_OS.dat", allow_pickle=True)
    coordinates_target = np.nonzero(target_O)

    # Discretize and set the recurrent connections
    sbs.bin_omega(C_real=np.copy(C), min=min_C, max=max_C)
    sbs.set_recurrent_connection()

    # Smooth the input to get X also calculate the rates of the target
    X = np.zeros((utils.Nx, utils.Ntime))
    R = np.zeros((utils.Nneuron, utils.Ntime))
    target_R = np.zeros((utils.Nneuron, utils.Ntime))
    for t in range(1, utils.Ntime):
        X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:, t]
        neurons_that_spiked = np.nonzero(target_O[:,t])[0]
        r_tmp = target_R[:,t-1]
        r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked))
        target_R[:,t] = (1-utils.lam*utils.dt)*r_tmp

    if(sbs.debug):
        plt.plot(X.T)
        plt.show()

    # Best so far is c_e7c_i5f_e255f_i200dup0.01000ddwn0.01000l2_49.47701
    max_cor = 0.0
    min_l2 = 1000000.0
    # min_l2 = 38.9

    delta_mod_ups = [0.01] # <
    delta_mod_dwns = [0.01] # >
    coarse_exc = [7] # <
    coarse_inh = [5] # <
    fine_exc = [255] # >
    fine_inh = [200] # <
    
    totalT = len(delta_mod_ups)*len(delta_mod_dwns)*len(coarse_exc)*len(coarse_inh)*len(fine_exc)*len(fine_inh)
    print("Estimated time is %dmin" % (int(totalT*1.1/60)))
    l = 0

    bar = ChargingBar('Finding Biases', max=totalT)
    for c_e in coarse_exc: #Lowest to highest.
        no_spike_first_trial = False
        for f_e in fine_exc: # Highest to lowest. If The first trial does not spike, we can break and return to the outer loop.
            for (idx_c_i, c_i) in enumerate(coarse_inh): # Lowest to highest.
                for (idx_f_i,f_i) in enumerate(fine_inh): # Lowest to highest.
                    for (idx_delta_mod_up,delta_mod_up) in enumerate(delta_mod_ups): # Lowest to highest
                        for (idx_delta_mod_dwn,delta_mod_dwn) in enumerate(delta_mod_dwns): # Highest to lowest
                            bar.next()
                            sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", f_e, c_e)
                            sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", f_i, c_i)

                            if(sbs.debug):
                                print("\nSetting EXC biases to (%d,%d)" % (f_e,c_e))
                                print("Setting INH biases to (%d,%d)" % (f_i,c_i))

                            # Load the signal
                            sbs.load_signal(np.copy(X), delta_mod_up, delta_mod_up)

                            try:
                                O_DYNAPS = sbs.execute()
                            except IndexError:
                                if((idx_c_i + idx_f_i + idx_delta_mod_up + idx_delta_mod_dwn) == 0):
                                    no_spike_first_trial = True
                                    # We can safely abort all inner loops.
                                    if(sbs.debug):
                                        print("No spikes. Abort!")
                                    break
                            else:
                                # Compute the rate vectors
                                for t in range(1, utils.Ntime):
                                    neurons_that_spiked = np.nonzero(O_DYNAPS[:,t])[0]
                                    r_tmp = R[:,t-1]
                                    r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked))
                                    R[:,t] = (1-utils.lam*utils.dt)*r_tmp

                                # For every neuron, compute the correlation coefficient or l2 norm
                                cors = []
                                for i in range(utils.Nneuron):
                                    """if(np.std(target_R[i,:]) == 0 or np.std(R[i,:]) == 0):
                                        mean_diff = np.mean(np.abs(target_R[i,:]-R[i,:]))
                                        cors.append(np.exp(-3*mean_diff))

                                    else:
                                        cors.append(np.corrcoef(target_R[i,:], R[i,:])[0,1])"""
                                    # Use l2 norm
                                    cors.append(np.linalg.norm(np.abs(target_R[i,:] - R[i,:]),2))

                                cors = np.asarray(np.abs([0 if np.isnan(v) else v for v in cors]))
                                total_cor = np.sum(cors) / utils.Nneuron
                                if(sbs.debug):
                                    print(total_cor)
                                    # if(total_cor > max_cor): # Uncomment for correlation coef.
                                if(total_cor < min_l2):
                                    # max_cor = total_cor # Uncomment for correlation coef.
                                    min_l2 = total_cor
                                    if(sbs.debug):
                                        print("T-UP: %.6f     T_DWN: %.6f    P. Corr.: %.6f" % (delta_mod_up, delta_mod_dwn, total_cor))
                                    var_name = ("Resources/Bias/c_e%dc_i%df_e%df_i%ddup%.5fddwn%.5fl2_%.5f.png" % (c_e,c_i,f_e,f_i,delta_mod_up,delta_mod_dwn, total_cor))

                                    #! Remeber bias and thresholds for delta moodulator
                                    coordinates_dyn = np.nonzero(O_DYNAPS)
                                    plt.figure(figsize=(18, 6))
                                    plt.scatter(coordinates_target[1], coordinates_target[0]+1, marker='o', s=0.5, c='g')
                                    plt.scatter(coordinates_dyn[1], coordinates_dyn[0]+1, marker='o', s=0.5, c='r')
                                    plt.ylim((0,sbs.num_neurons+1))
                                    plt.yticks(ticks=np.linspace(0,sbs.num_neurons,int(sbs.num_neurons/2)+1))
                                    plt.savefig(var_name)
                                    if(sbs.debug):
                                        plt.show()
                        if(no_spike_first_trial):
                            break
                    if(no_spike_first_trial):
                        break
                if(no_spike_first_trial):
                    break
            if(no_spike_first_trial):
                break

    bar.finish()




def Learning(sbs, utils, F, FtM, C, debug = False):
    print("Setting FF...")
    sbs.F = np.copy(FtM)
    max_C = 0.412
    min_C = -0.412
    
    # Setting the weights on DYNAPS
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 200, 5)

    # Total training time
    TotTime = utils.Nit*utils.Ntime
    # Copy the initial recurrent weights
    Ci = np.copy(C)
    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Cs_discrete = np.zeros([utils.T, utils.Nneuron, utils.Nneuron])
    C_current_discrete = sbs.bin_omega(C_real=np.copy(C), min=min_C, max=max_C)

    delta_mod_thresh_up = 0.01 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.01
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
            # 7) #! In the simulation, we only update the rate vec at one point per time step
            for t in range(1, utils.Ntime):
                neurons_that_spiked = np.nonzero(O_DYNAPS[:,t])[0]
                r_tmp = R[:,t-1]
                r_tmp[neurons_that_spiked] += np.ones(len(neurons_that_spiked))
                R[:,t] = (1-utils.lam*utils.dt)*r_tmp
            # 8)
            for t in range(1, utils.Ntime):
                current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
                new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, X[:,t]) + np.matmul(C, R[:,t-1])

                (m, k) = my_max(new_V_recon.reshape((-1,1)) - current_thresh) # Returns maximum and argmax
                if(m[0] >= 0):
                    tmp = utils.epsr*(utils.beta*(new_V_recon + utils.mu*R[:,t-1]) + C[:,k] + utils.mu*Id[:,k])
                    delta_Omega[:,k] += tmp
                    ks[k] += 1
                V_recons[:,t] = new_V_recon

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