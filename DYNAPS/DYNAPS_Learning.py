import numpy as np
import matplotlib.pyplot as plt
from DYNAPS_runnet import runnet, my_max
from progress.bar import ChargingBar
from metrics.kreuz import distance as kreuz_distance

# TODO - Write code for coordinate-descent, 
# TODO - Use different signal on-chip than off-chip and (hopefully) show that reconstruction does not work


def compare_population_kreuz(OT_1, OT_2):
    Nneurons = OT_1.shape[0]
    Ntime = OT_1.shape[1]
    if(not Ntime == OT_2.shape[1]):
        raise AssertionError("Spike trains don't have the same length")

    similarities = np.zeros((Nneurons, Ntime))

    for i in range(Nneurons):
        st_one = np.nonzero(OT_1[i,:])[0]
        st_two = np.nonzero(OT_2[i,:])[0]
        t, dist = kreuz_distance(st_one,st_two,start=0,end=Ntime,nsamples=Ntime)
        similarities[i,:] = dist
    
    return similarities, np.sum(similarities)/(Nneurons*Ntime)

def compare_population_von_rossum(OT_1,OT_2):
    lam = 50
    dt = 0.001
    Nneurons = OT_1.shape[0]
    Ntime = OT_1.shape[1]
    if(not Ntime == OT_2.shape[1]):
        raise AssertionError("Spike trains don't have the same length")
    R_1 = np.zeros((Nneurons,Ntime))
    R_2 = np.zeros((Nneurons,Ntime))
    error = np.zeros(Ntime)
    for t in range(1,Ntime):
        R_1[:,t] = (1-lam*dt)*(R_1[:,t-1] + OT_1[:,t])
        R_2[:,t] = (1-lam*dt)*(R_2[:,t-1] + OT_2[:,t])
        error[t] = np.sum(np.abs(R_1[:,t]-R_2[:,t])) / Nneurons

    return np.sum(error)/Ntime


def tune_biases(sbs, utils, metric = 'kreuz'):

    try:
        F_disc = np.load("Resources/DYNAPS/DYNAPS_F_disc.dat", allow_pickle=True)
        C_cont = np.load("Resources/DYNAPS/DYNAPS_C_cont.dat", allow_pickle=True)
        OT_sim = np.load("Resources/DYNAPS/OT_sim.dat", allow_pickle=True)
        X = np.load("Resources/DYNAPS/bias_xT.dat",allow_pickle=True)
    except:
        print("Error: Failed to load resources.")
        return

    OT_sim = OT_sim[:,0:utils.Ntime]
    X = X[:,0:utils.Ntime]

    # 1) Set F_disc
    sbs.F = np.copy(F_disc)
    max_C = 0.4
    min_C = -0.4

    if(sbs.debug):
        plt.figure(figsize=(12,5))
        coordinates = np.nonzero(OT_sim)
        plt.scatter(coordinates[1], coordinates[0]+1, s=0.5, color='g', marker='o')
        plt.show()

    sbs.bin_omega(C_cont,min_C,max_C)
    sbs.set_recurrent_connection()

    num_parameters = 5
    fine_inh = [255,200,100,50,20]
    coarse_inh = [4,5,6,7,-1]
    delta_mod_ups = [0.05,0.01,0.005,0.001,0.0005]
    delta_mod_dwns = [0.05,0.01,0.005,0.001,0.0005]
    best = np.inf

    best_fi = fine_inh[0]
    best_ci = coarse_inh[0]
    best_dup = delta_mod_ups[0]
    best_ddwn = delta_mod_dwns[0]

    errors = []
    OT_DYNAPS_initial = np.zeros((utils.Nneuron,utils.Ntime))
    OT_DYNAPS_best = np.zeros((utils.Nneuron,utils.Ntime))

    order = ['ci', 'fi', 'dup', 'ddwn']
    num_swipes = 1
    T = 4
    for t in range(T*num_swipes):
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
            sbs.load_signal(X, delta_mod_thresh_up, delta_mod_thresh_dwn)

            # 3)
            _,OT_DYNAPS,_ = runnet(sbs,utils,-1,-1,-1,X.shape[1],-1,find_bias=True)
            if(t == 0 and i == 0):
                OT_DYNAPS_initial = np.copy(OT_DYNAPS)

            if(metric == 'kreuz'):
                _ , error = compare_population_kreuz(OT_DYNAPS,OT_sim)
            elif(metric == 'vonRossum'):
                error = compare_population_von_rossum(OT_DYNAPS,OT_sim)
            
            errors.append(error)

            if(error < best):
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
                best = error
                # errors.append(error)
                OT_DYNAPS_best = np.copy(OT_DYNAPS)

                if(sbs.debug):
                    plt.figure(figsize=(12,5))
                    coordinates = np.nonzero(OT_sim)
                    plt.scatter(coordinates[1], coordinates[0]+1, s=0.5, color='g', marker='o', label="Simulation")
                    coordinates = np.nonzero(OT_DYNAPS)
                    plt.scatter(coordinates[1], coordinates[0]+1, s=0.5, color='r', marker='o', label="DYNAPS")
                    ax = plt.gca()
                    ax.set_xticks([],[]); ax.set_yticks([],[])
                    ax.legend()
                    plt.show()

                print("Fi: %d Ci: %d Dup: %.4f Ddwn: %.4f Error: %.4f" % (best_fi,best_ci,best_dup,best_ddwn, error))

    errors = np.asarray(errors)
    errors.dump(("Resources/DYNAPS/%s_errors.dat" % metric))

    ########## Plotting ################
    if(metric == 'kreuz'):
        name = 'Kreuz'
    else:
        name = "Von Rossum"

    title_font_size = 6
    axis_font_size = 5
    ticks_font_size = 5
    linewidth = 0.5

    color_true = 'C1'
    color_recon = 'C2'
    markersize = 0.5 #! Change to 0.00001
    marker = 'o' #! Change to ,

    try:
        errors_von_rossum = np.load("Resources/DYNAPS/vonRossum_errors.dat", allow_pickle=True)
        errors_kreuz = np.load("Resources/DYNAPS/kreuz_errors.dat", allow_pickle=True)
        if(not len(errors_kreuz) == len(errors_von_rossum)):
            raise Exception("Error lengths do not match")
    except:
        print("Failed to load kreuz and von rossum errors")
    else:
        plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax2 = ax.twinx()
        ax.plot(errors_kreuz, color=color_true, linewidth=linewidth)
        ax.set_xlabel('Iteration', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Kreuz distance', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.plot(errors_von_rossum, color=color_recon, linewidth=linewidth)
        ax2.set_xlabel('Iteration', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax2.set_ylabel('Von Rossum distance', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax2.tick_params(axis='x', labelsize=ticks_font_size)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)
        plt.savefig(("Resources/DYNAPS/%s_coordinate_descent.eps" % metric), format='eps')
        plt.show()

    plt.figure(figsize=(12,6))
    plt.subplot(211)
    plt.title('Alignment using initial biases', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    coordinates = np.nonzero(OT_sim)
    ax.scatter(coordinates[1], coordinates[0]+1, s=markersize, color=color_true, marker=marker, label="Simulation")
    coordinates = np.nonzero(OT_DYNAPS_initial)
    plt.scatter(coordinates[1], coordinates[0]+1, s=markersize, color=color_recon, marker=marker, label="DYNAPS")
    ax = plt.gca()
    ax.set_xticks([],[]); ax.set_yticks([],[])
    L = ax.legend()
    plt.setp(L.texts, family='Times New Roman',fontsize=5)

    plt.subplot(212)
    plt.title('Alignment using best biases', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    coordinates = np.nonzero(OT_sim)
    plt.scatter(coordinates[1], coordinates[0]+1, s=markersize, color=color_true, marker=marker)
    coordinates = np.nonzero(OT_DYNAPS_best)
    plt.scatter(coordinates[1], coordinates[0]+1, s=markersize, color=color_recon, marker=marker)
    ax.set_xticks([],[]); ax.set_yticks([],[])

    plt.tight_layout()
    plt.savefig(("Resources/DYNAPS/%s_bias_pre_and_post_descent.eps" % metric), format='eps')
    plt.show()



def Learning(sbs, utils, F, FtM, C, debug = False):
    print("Setting FF...")
    sbs.F = np.copy(FtM)
    max_C = 0.625
    min_C = -0.545
    
    ########################### Biases producing reasonable learning @270 iterations ##############
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 255, 5)

    delta_mod_thresh_up = 0.05 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.05

    ########################### Biases from coordinate-descent-approach ###########################
    """sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 20, 7)

    delta_mod_thresh_up = 0.0005 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.05"""

    # Total training time
    TotTime = utils.Nit*utils.Ntime
    # Copy the initial recurrent weights
    Ci = np.copy(C)
    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Cs_discrete = np.zeros([utils.T, utils.Nneuron, utils.Nneuron])
    C_current_discrete = sbs.bin_omega(C_real=np.copy(C), min=min_C, max=max_C)

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

    # Generate distribution of DYNAPS-simulation spike time mismatch
    spike_alignment_distribution = []

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
            # 7) + 8)

            for t in range(1, utils.Ntime):
                current_thresh = utils.Thresh-0.01*np.random.randn(utils.Nneuron, 1)
                
                # new_V_recon = 0.1*V_recons[:,t-1] + np.matmul(F.T, X[:,t]) + np.matmul(C, R[:,t-1]) + 0.001*np.random.randn(utils.Nneuron, 1).ravel()
                new_V_recon = np.matmul(F.T, X[:,t]) + np.matmul(C, R[:,t-1]) + 0.001*np.random.randn(utils.Nneuron, 1).ravel()

                (m, k) = my_max(new_V_recon.reshape((-1,1)) - current_thresh) # Returns maximum and argmax
                has_spike = np.sum(O_DYNAPS[k,max(0,t-utils.alignment_delta_t):min(utils.Ntime-1,t+utils.alignment_delta_t)]) > 0

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

    xestc_initial = np.zeros((utils.Nx, TimeT))
    xestc_after = np.zeros((utils.Nx, TimeT))
    X_save = np.zeros((utils.Nx, TimeT))
    O_DYNAPS_initial = np.zeros((utils.Nneuron, TimeT))
    O_DYNAPS_after = np.zeros((utils.Nneuron, TimeT))

    Trials = 5
    bar = ChargingBar(('Errors'), max=Trials*utils.T)
    for r in range(Trials):
        InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T

        for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

        if(r == 0):
            X_save = np.copy(xT)
        
        # Load the input
        sbs.load_signal(np.copy(xT), delta_mod_thresh_up, delta_mod_thresh_dwn)

        for i in range(utils.T):
            if(sbs.debug):
                print(Cs_discrete[i,:,:])
            
            (rOT, OT, VT) = runnet(sbs, utils, F, Cs_discrete[i,:,:], Cs[i,:,:], TimeT, xT)
            xestc = np.matmul(Decs[i,:,:], rOT) # Decode the rate vector
            err = np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)

            if(r == 0 and i == 0):
                xestc_initial = np.copy(xestc)
                O_DYNAPS_initial = OT
            if(r == 0 and i == utils.T-1):
                xestc_after = np.copy(xestc)
                O_DYNAPS_after = OT

            #if(sbs.debug):
            print("Error is %.3f  Absolute weights: %d" % (err, np.sum(np.abs(Cs_discrete[i,:,:]))))           
            
            Error[0,i] = Error[0,i] + err
            MeanPrate[0,i] = MeanPrate[0,i] + np.sum(OT) / (TimeT*utils.dt*utils.Nneuron*Trials)
            MembraneVar[0,i] = MembraneVar[0,i] + np.sum(np.var(VT, axis=1, ddof=1)) / (utils.Nneuron*Trials)
            bar.next()
    bar.next(); bar.finish()

    ErrorC = np.zeros((1,utils.T))
    for i in range(utils.T):
        CurrC = Cs[i,:,:]

        Copt = np.matmul(-F.T, F)
        optscale = np.trace(np.matmul(CurrC.T, Copt)) / np.sum(Copt**2)
        Cnorm = np.sum(CurrC**2)
        ErrorC[0,i] = np.sum(np.sum((CurrC - optscale*Copt)**2 ,axis=0)) / Cnorm

    ending = ("_%s" % str(sbs.parameters["alignment_delta_t"]))

    return_dict = {
        "Fi": F,
        "F_dis": FtM,
        "Ci": Ci,
        "Deci": Decs[0,:,:],
        "F_after": F,
        "C_after": C,
        "D_after": Dec,
        "Cs": Cs,
        "Cs_discrete": Cs_discrete,
        "Decs": Decs,
        ("DYNAPS_Error%s" % ending): Error,
        ("DYNAPS_MeanPrate%s" % ending): MeanPrate,
        ("DYNAPS_MembraneVar%s" % ending): MembraneVar,
        ("DYNAPS_ErrorC%s" % ending): ErrorC,
        "w": w,
        "DYNAPS_xestc_initial": xestc_initial,
        "DYNAPS_xestc_after": xestc_after,
        "O_DYNAPS_initial": O_DYNAPS_initial,
        "O_DYNAPS_after": O_DYNAPS_after,
        "DYNAPS_xT": X_save
        }

    return return_dict


def run_testing(sbs, utils):

    # Setting the weights on DYNAPS
    sbs.groups[4].set_bias("PS_WEIGHT_EXC_F_N", 255, 7)
    sbs.groups[4].set_bias("PS_WEIGHT_INH_F_N", 255, 5)

    delta_mod_thresh_up = 0.05 # Use the one in sbs controller
    delta_mod_thresh_dwn = 0.05

    C_i = np.load("Resources/DYNAPS/Ci.dat", allow_pickle=True)
    C_after = np.load("Resources/DYNAPS/C_after.dat", allow_pickle=True)
    Di = np.load("Resources/DYNAPS/Deci.dat", allow_pickle=True)
    D_after = np.load("Resources/DYNAPS/D_after.dat", allow_pickle=True)
    F = np.load("Resources/DYNAPS/Fi.dat", allow_pickle=True)
    Cs_discrete = np.load("Resources/DYNAPS/Cs_discrete.dat", allow_pickle=True)

    C_i_dis = Cs_discrete[0,:,:]
    C_after_dis = Cs_discrete[utils.T-1,:,:]
    F_dis = np.load("Resources/DYNAPS/F_dis.dat", allow_pickle=True)

    sbs.F = F_dis

    # Generate a signal
    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)

    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

    X = np.zeros((utils.Nx, utils.Ntime))
    for t in range(1, utils.Ntime):
        X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:, t]

    sbs.load_signal(X, delta_mod_thresh_up, delta_mod_thresh_dwn)

    # Initial run
    R_i, O_DYNAPS_i, V_recons_i = runnet(sbs, utils, F, C_i_dis, C_i, utils.Ntime, X, plot_dist=True)
    
    # After learning
    R_after, O_DYNAPS_after, V_recons_after = runnet(sbs, utils, F, C_after_dis, C_after, utils.Ntime, X, plot_dist=True)

    # Estimate X
    xest_initial = np.matmul(Di, R_i)
    xest_after = np.matmul(D_after, R_after)

    plt.figure(figsize=(16, 10))
    subplot = 611
    plt.title('Initial reconstruction (green) of the target signal (red)')
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Initial reconstruction (green) of the target signal (red)')
        plt.plot(X[i,:], 'r')
        plt.plot(xest_initial[i,:], 'g')
        subplot = subplot+1
    
    # Plot initial spike trains
    plt.subplot(subplot)
    plt.title('Initial spike trains')
    coordinates_intial = np.nonzero(O_DYNAPS_i)
    plt.scatter(coordinates_intial[1], coordinates_intial[0], s=0.8, marker="o", c="k")
    subplot = subplot+1

    # Plot after learning
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Post-learning reconstruction (green) of the target signal (red)')
        plt.plot(X[i,:], 'r')
        plt.plot(xest_after[i,:], 'g')
        subplot = subplot+1

    # Plot post-learning spike trains
    plt.subplot(subplot)
    plt.title('Post-learning spike trains')
    coordinates_after = np.nonzero(O_DYNAPS_after)
    plt.scatter(coordinates_after[1], coordinates_after[0], s=0.8, marker="o", c="k")
    subplot = subplot+1

    plt.tight_layout()
    plt.savefig("Resources/after_training.png")
    plt.show()
