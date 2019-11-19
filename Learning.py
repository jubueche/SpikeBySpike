import numpy as np
import matplotlib.pyplot as plt
from Utils import my_max
from runnet import *
import warnings
from progress.bar import ChargingBar
import os


def discretize(Omega, number_of_bins):
    C = np.copy(Omega)

    diag_C = np.diagonal(C)
    np.fill_diagonal(C,0)
    C_flat = C.ravel()

    c_std = np.std(C_flat, ddof=1)
    if(np.isnan(c_std)):
        return Omega
    c_mean = np.mean(C_flat)

    left_68 = c_mean - c_std
    right_68 = c_mean + c_std
    num_middle = int(number_of_bins*0.68)
    num_rest_half = int(0.5* (number_of_bins - num_middle))

    inner_bin_edges = np.linspace(left_68, right_68, num=num_middle)
    outer_bin_edges_left = np.linspace(c_mean - 3*c_std, c_mean - c_std, num=num_rest_half)[0:num_rest_half-1]
    outer_bin_edges_right = np.linspace(c_mean + c_std, c_mean + 3*c_std, num=num_rest_half)[1:]

    bin_edges = np.hstack([outer_bin_edges_left,inner_bin_edges,outer_bin_edges_right])

    indices = np.digitize(C_flat, bins = bin_edges, right = True)

    def round_to_nearest(x,l,r):
        if(round((x-l)/(r-l)) == 0):
            return l
        else:
            return r
    n = len(bin_edges)
    for idx,i in enumerate(indices):
        if(i >= n):
            i = n-1
        C_flat[idx] = round_to_nearest(C_flat[idx], bin_edges[i-1], bin_edges[i])

       
    C = C_flat.reshape(C.shape)
    np.fill_diagonal(C, diag_C)

    return C


def discretize_linear(Omega, number_of_bins):
    C = np.copy(Omega)

    diag_C = np.diagonal(C)
    np.fill_diagonal(C,0)
    C_flat = C.ravel()
    c_min = np.min(C_flat)
    c_max = np.max(C_flat)

    bin_edges = np.linspace(c_min, c_max, number_of_bins)

    indices = np.digitize(C_flat, bins = bin_edges, right = True)

    def round_to_nearest(x,l,r):
        if(round((x-l)/(r-l)) == 0):
            return l
        else:
            return r
    n = len(bin_edges)
    for idx,i in enumerate(indices):
        if(i >= n):
            i = n-1
        C_flat[idx] = round_to_nearest(C_flat[idx], bin_edges[i-1], bin_edges[i])

    C = C_flat.reshape(C.shape)
    np.fill_diagonal(C, diag_C)

    return C

def get_input(duration, utils, w, audio_helper = None, use_audio = False, training = True, digit = -1):

    if(not use_audio):
        Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), duration)).T
        Input[:,0:100] = 0
        for d in range(utils.Nx):
            Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

        return Input

    else:
        if(training):
            label, Input = audio_helper.get_next_training(digit)
        else:
            label, Input = audio_helper.get_next_test(digit)
        

        Input = Input.reshape((utils.Nx,-1))
        for d in range(utils.Nx):
            Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

        """plt.plot(Input.T)
        plt.show()"""

        X = np.zeros(Input.shape)
        for t in range(Input.shape[1]):
            X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:,t]
        
        
        """plt.plot(X.T)
        plt.show()"""

        return label, Input


def get_reward(utils, C, F, w, use_audio = False, use_spiking = False, audio_helper = None):

    # Generate input and integrate
    if(not use_audio):
        TimeL = 1000
    else:
        TimeL = utils.Ntime

    xL = np.zeros((utils.Nx, TimeL))

    # Generate new input
    if(use_audio):
        label, InputL = get_input(TimeL, utils, w, audio_helper=audio_helper, use_audio = use_audio,training=True)
    else:
        InputL = get_input(TimeL, utils, w, audio_helper=audio_helper, use_audio = use_audio,training=True)

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    # Obtain Decoder for current recurrent matrix
    (rOL,_,_) = runnet(utils,utils.dt, utils.lam, F, InputL, C, utils.Nneuron, TimeL, utils.Thresh, use_spiking=use_spiking)
    Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T

    # Now test on new input using Dec
    if(not use_audio):
        TimeT = 10000
    else:
        TimeT = utils.Ntime

    Error = 0
    MeanPrate = 0
    xT = np.zeros((utils.Nx, TimeT))

    Trials = 5

    for r in range(Trials):

        if(use_audio):
            label, InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False)
        else:
            InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False)

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

        (rOT, OT, VT) = runnet(utils,utils.dt, utils.lam, F, InputT, C, utils.Nneuron, TimeT, utils.Thresh,use_spiking=use_spiking)
        xestc = np.matmul(Dec, rOT) # Decode the rate vector
        Error = Error + np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)
        MeanPrate = MeanPrate + np.sum(OT) / (TimeT*utils.dt*utils.Nneuron*Trials) 
    print("Mean firing rate: %.3f" % MeanPrate)  
    return Error, MeanPrate

def update_lookback(lookback, new_val):
    lookback_return = np.zeros(len(lookback))
    lookback_return[1:] = lookback[0:len(lookback)-1]
    lookback_return[0] = new_val
    return lookback_return

def look_back_reward(lookback):
    lam = 1
    reward = 0
    s = 0
    for i in range(len(lookback)):
        reward += lookback[i]*lam**(-i)
        s += lam**(-i)
    return reward / s

def Learning(utils, F, C, update_all = False, discretize_weights = False, number_of_bins = 100,
            remove_positive = False, use_spiking = False, use_batched=False, use_batched_nn=False,
            use_audio=False, audio_helper = None, use_reinforcement = False):

    if(use_audio and audio_helper is None):
        raise Exception("Audio helper is None")

    initial_reward = 0
    look_back_length = 5
    reward_lookback = np.zeros(look_back_length) # First element is most recent and has highest weight.
    scaled_rewards = []
    alpha = utils.epsr
    TotTime = utils.Nit*utils.Ntime

    Fi = np.copy(F)
    if(discretize_weights):
        C = discretize_linear(C, number_of_bins)
    Ci = np.copy(C)

    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Fs = np.zeros([utils.T, utils.Nx, utils.Nneuron]) # Store the FF weights over the course of training

    V = np.zeros((utils.Nneuron, 1))
    Ca = np.zeros((utils.Nneuron, 1))
    Cas = np.zeros((utils.Nneuron, utils.Ntime))

    O = 0
    k = 0 #! Indexing starts with 0
    r0 = np.zeros((utils.Nneuron, 1))
    ot = np.zeros((utils.Nneuron, 1))

    x = np.zeros((utils.Nx, 1))
    
    ### For spiking input ###
    I = np.zeros((2*utils.Nx, 1)) # Input current
    X = np.zeros((utils.Nx, utils.Ntime))
    M = np.asarray([[1,-1,0,0],[0,0,1,-1]])
    x_recon_lam = 0.001
    x_recon_R = 40.3
    V_ftmi = np.zeros((utils.Nneuron, 1))
    V_ftmis = np.zeros((utils.Nneuron,utils.Ntime))
    Vs = np.zeros((utils.Nneuron,utils.Ntime))
    FTMIS = np.zeros((utils.Nneuron,utils.Ntime))
    FtI = np.zeros((utils.Nneuron,utils.Ntime))
    ### End For spiking input ###

    ### For batched updates ###
    # Save the updates in here
    batched_delta_Omega = np.zeros(C.shape)
    # Keep track of how many times C[:,k] was updated
    ks = np.zeros(C.shape[1])
    ### End for batched update ###

    Input = np.zeros((utils.Nx, utils.Ntime))
    Id = np.eye(utils.Nneuron)

    if(not use_audio):
        w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
        w = w / np.sum(w)
    else:
        w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,500,500)-250)**2)/(2*utils.sigma**2))
        w = w / np.sum(w)

    j = 1
    num_update = 0
    discount_step = 10
    discount_rate = 0.7

    bar = ChargingBar('Learning', max=TotTime-1)
    for i in range(2, TotTime):

        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = C # Indexing starts at 0
            Fs[j-1,:,:] = F
            j = j+1

        if(((i-2) % utils.Ntime) == 0):
            num_update += 1

            if(use_audio):
                _, Input = get_input(utils.Ntime, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=True)
            else:
                Input = get_input(utils.Ntime, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=True)

            # Convert to spikes
            if(use_spiking):
                (OT_down, OT_up) = get_spiking_input(30, Input, utils.Nx, utils.Ntime)
                I = 0
                ot *= 0

            if(use_batched):
                if(not use_batched_nn):
                    for i in range(batched_delta_Omega.shape[1]): # First transform each column
                        if(ks[i] > 0):
                            batched_delta_Omega[:,i] /= ks[i]
                # Do the update
                C = C + batched_delta_Omega

                # Reset
                ks = np.zeros(batched_delta_Omega.shape[1])
                batched_delta_Omega = np.zeros(C.shape)

            # Integrate back for sanity check
            """
            for t in range(1,utils.Ntime):
                X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:,t]
            
            # Convert to spikes
            (OT_down, OT_up) = get_spiking_input(0.5, X, utils.Nx, utils.Ntime)

            X_recon = np.zeros(X.shape)
            for t in range(1,utils.Ntime):
                ot_in = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
                X_recon[:,t] = (1-0.001)*X_recon[:,t-1] + 0.55*np.matmul(M,ot_in).ravel()
            plt.figure(figsize=(6.00,1.57))
            plt.title('True input (blue) versus reconstructed input (orange)', fontname="Times New Roman" ,fontsize=6)
            plt.xticks([],[]); plt.yticks([],[])
            plt.plot(X[0,:].T, linewidth=0.5)
            plt.plot(X_recon[0,:].T, linewidth=0.5)
            plt.savefig("Data/spiking_reconstructed.eps", format="eps")
            plt.show()"""

            """if (not i == 2):  
                plt.plot(Vs[0,:])
                plt.plot(V_ftmis[0,:])
                plt.show()"""

            """if (not i == 2):
                plt.plot(FtI[0,:])
                plt.plot(FTMIS[0,:])
                plt.show()"""
        
        t = ((i-2) % utils.Ntime)

        if(use_spiking):
            ot_in = np.asarray([OT_up[0,t], OT_down[0,t], OT_up[1,t], OT_down[1,t]]).reshape((-1,1))
            I = (1-x_recon_lam)*I + x_recon_R*ot_in
            FTMI = np.matmul(np.matmul(F.T, M), I)
            FTMIS[:,t] = FTMI.ravel()
            FtI[:,t] = np.matmul(F.T, Input[:,(i % utils.Ntime)].reshape((-1,1))).ravel()
            V = (1-utils.lam*utils.dt)*V + utils.dt*FTMI.reshape((-1,1)) + np.matmul(C,ot).reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
            V_ftmis[:,t] = V.ravel()
        else:
            ### The true V
            V = (1-utils.lam*utils.dt)*V + utils.dt*np.matmul(F.T, Input[:,(i % utils.Ntime)].reshape((-1,1))) + np.matmul(C,ot).reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
            Vs[:,t] = V.ravel()

        x = (1-utils.lam*utils.dt)*x + utils.dt*Input[:, (i % utils.Ntime)].reshape((-1,1))


        (m, k) = my_max(V - utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)) # Returns maximum and argmax
        #! Use multiple spikes for the update
        ot = np.zeros((utils.Nneuron,1))
        if(update_all):
            ot[((V - utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)) >= 0).ravel()] = 1.0
        elif(m >= 0):
            ot[k] = 1.0
        
        # Update the Calcium variable
        #Ca = 0.7*Ca + np.matmul(C,ot).reshape((-1,1)) # Let the spike already propagate over the recurrent synapses.
        Ca = 0.2*V + np.matmul(C,ot).reshape((-1,1)) + utils.dt*np.matmul(F.T, Input[:,((i+1) % utils.Ntime)].reshape((-1,1)))
        Cas[:,t] = Ca.reshape((-1,))

        # Use whole spike vector
        #delta_Omega = - utils.epsr*(utils.beta*np.matmul(V + utils.mu*r0, ot.T) + np.matmul(C + utils.mu*Id,np.matmul(ot,ot.T)))
        delta_Omega = - utils.epsr*(utils.beta*np.matmul(Ca + utils.mu*r0, ot.T) + np.matmul(C + utils.mu*Id,np.matmul(ot,ot.T)))
        if(use_batched):
            if(update_all):
                batched_delta_Omega[:,ot.astype(bool).ravel()] += delta_Omega[:,ot.astype(bool).ravel()]
                ks[ot.astype(bool).ravel()] += 1
            else:
                batched_delta_Omega[:,k] += delta_Omega[:,k]
                ks[k] += 1
        else:
            C = C + delta_Omega
        
        if(discretize and remove_positive):
            C[C > 0] = 0.0
            C = discretize_linear(C, number_of_bins)

        elif(discretize_weights):
            C = discretize_linear(C, number_of_bins)

        elif(remove_positive):
            C[C > 0] = 0.0

        if(remove_positive):
            assert (C <= 0).all(), "Positive values in C encountered"

        """if(num_update == discount_step):
            num_update = 0
            tmp = np.copy(utils.epsr)
            utils.epsr *= discount_rate
            print("Updated learning rate from %.4f to %.4f" % (tmp, utils.epsr))"""
        
        if(use_reinforcement and (((i-2) % 250) == 0)):
            reward, MeanPrate = get_reward(utils, C, F, w, use_audio, use_spiking, audio_helper)
            reward_lookback = update_lookback(reward_lookback, reward)
            reward = look_back_reward(reward_lookback)
            if(i < look_back_length*250): # Burn in phase
                initial_reward = reward # Use initial reward as a reference for further rewards
                print("Initial reward is %.4f" % initial_reward)
            else:
                scaled_reward = reward / initial_reward
                if(scaled_reward < 1.0):
                    pass
                    #scaled_reward = scaled_reward**10 # Conservative
                #corrected = alpha * scaled_reward
                #print("Reward is %.3f Scaled reward is %.3f EpsR_prior is %.5f Corrected_Eps_R is %.5f" % (reward, scaled_reward, utils.epsr, corrected))
                print("Reward is %.3f" % reward)
                #utils.epsr = corrected # Use for reinforcement
                scaled_rewards.append(reward)

        r0 = r0 + ot
        
        r0 = (1-utils.lam*utils.dt)*r0
        bar.next()
    bar.next()
    bar.finish()
    if(use_reinforcement):
        scaled_rewards = np.asarray(scaled_rewards)
        ending = ("no_reinforcement%d_%d.dat" % (discount_step, discount_rate))
        scaled_rewards.dump(("scaled_rewards_adaptive_learning_%s" % ending))
        
    ########## Compute the optimal decoder ##########
    if(not use_audio):
        TimeL = 50000
    else:
        TimeL = utils.Ntime

    xL = np.zeros((utils.Nx, TimeL))
    Decs = np.zeros([utils.T, utils.Nx, utils.Nneuron])

    # Generate new input
    if(use_audio):
        label, InputL = get_input(TimeL, utils, w, audio_helper=audio_helper, use_audio = use_audio,training=True)
    else:
        InputL = get_input(TimeL, utils, w, audio_helper=audio_helper, use_audio = use_audio,training=True)

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    print(""); bar = ChargingBar('Decoders', max=utils.T)
    for i in range(utils.T):
        if(remove_positive):
            assert (Cs[i,:,:] <= 0).all(), "Encountered positive values in C"
        (rOL,_,_) = runnet(utils,utils.dt, utils.lam, Fs[i,:,:], InputL, Cs[i,:,:], utils.Nneuron, TimeL, utils.Thresh, use_spiking=use_spiking)
        Dec = np.linalg.lstsq(rOL.T, xL.T, rcond=None)[0].T # Returns solution that solves xL = Dec*r0L
        Decs[i,:,:] = Dec
        bar.next()
    bar.next(); bar.finish()

    if(not use_audio):
        TimeT = 10000
    else:
        TimeT = utils.Ntime

    MeanPrate = np.zeros((1,utils.T))
    Error = np.zeros((1,utils.T))
    MembraneVar = np.zeros((1,utils.T))
    xT = np.zeros((utils.Nx, TimeT))
    X_save = np.zeros((utils.Nx, TimeT))
    OT_sim = np.zeros((utils.Nneuron, TimeT))

    Trials = 10

    print(""); bar = ChargingBar('Errors', max=Trials*utils.T)
    for r in range(Trials):

        if(use_audio):
            label, InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False)
        else:
            InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False)

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

        for i in range(utils.T):
            if(remove_positive):
                assert (Cs[i,:,:] <= 0).all(), "Encountered positive values in C"
            (rOT, OT, VT) = runnet(utils,utils.dt, utils.lam, Fs[i,:,:], InputT, Cs[i,:,:], utils.Nneuron, TimeT, utils.Thresh,use_spiking=use_spiking)
            xestc = np.matmul(Decs[i,:,:], rOT) # Decode the rate vector
            Error[0,i] = Error[0,i] + np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)
            MeanPrate[0,i] = MeanPrate[0,i] + np.sum(OT) / (TimeT*utils.dt*utils.Nneuron*Trials)
            mvar = np.var(VT, axis=1, ddof=1)
            for idx,m in enumerate(mvar):
                mvar[idx] = min(500,m)
            MembraneVar[0,i] = MembraneVar[0,i] + np.sum(mvar) / (utils.Nneuron*Trials)

            if(r == 0 and i == utils.T-1):
                X_save = np.copy(xT)
                OT_sim = np.copy(OT)

            bar.next()
        bar.next()
    bar.next(); bar.finish()


    """if(use_audio):
        # Generate a training set and test set from this
        data_directory = os.path.join(os.getcwd(), "DYNAPS/Resources/Simulation/Dataset/")
        if(not os.path.exists(data_directory)):
            os.mkdir(data_directory)

        xT = np.zeros((utils.Nx, utils.Ntime))
        Data_training = np.zeros([audio_helper.train_number,utils.Nx, utils.Ntime])
        Data_testing = np.zeros([audio_helper.test_number,utils.Nx, utils.Ntime])
        label_training = np.zeros(audio_helper.train_number)
        label_testing = np.zeros(audio_helper.test_number)
        c_training = 0; c_testing = 0
        for i in range(10):
            for j in range(len(audio_helper.data_train[i])):
                label, Input = get_input(utils.Ntime, utils, w, audio_helper=audio_helper, use_audio=True, training=True,digit=i)

                for t in range(1,utils.Ntime):
                    xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*Input[:,t-1]

                (rOT, OT, _) = runnet(utils,utils.dt, utils.lam, F, Input, C, utils.Nneuron, utils.Ntime, utils.Thresh,use_spiking=use_spiking)
                xest = np.matmul(Dec, rOT)
                Data_training[c_training,:,:] = xest
                #Data_training[c_training,:,:] = rOT
                label_training[c_training] = label
                c_training += 1

            for j in range(len(audio_helper.data_test[i])):
                label, Input = get_input(utils.Ntime, utils, w, audio_helper=audio_helper, use_audio=True, training=False,digit=i)

                for t in range(1,utils.Ntime):
                    xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*Input[:,t-1]

                (rOT, OT, _) = runnet(utils,utils.dt, utils.lam, F, Input, C, utils.Nneuron, utils.Ntime, utils.Thresh,use_spiking=use_spiking)
                xest = np.matmul(Dec, rOT)
                Data_testing[c_testing,:,:] = xest
                label_testing[c_testing] = label
                c_testing += 1

        Data_testing.dump(os.path.join(data_directory, "Testing_spikes.dat"))
        Data_training.dump(os.path.join(data_directory, "Training_spikes.dat"))
        label_training.dump(os.path.join(data_directory, "Training_labels.dat"))
        label_testing.dump(os.path.join(data_directory, "Testing_labels.dat"))"""
        

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
        "w": w,
        "OT_sim": OT_sim,
        "xT": X_save
        }

    return return_dict



