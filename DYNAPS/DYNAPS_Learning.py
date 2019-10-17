import numpy as np
import matplotlib.pyplot as plt
from DYNAPS_runnet import runnet, my_max
from progress.bar import ChargingBar

def Learning(sbs, utils, F, FtM, C, debug = False):
    print("Setting FF...")

    sbs.F = np.copy(FtM)
    
    # Total training time
    TotTime = utils.Nit*utils.Ntime
    # Copy the initial recurrent weights
    Ci = np.copy(C)
    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Cs_discrete = np.zeros([utils.T, utils.Nneuron, utils.Nneuron])
    C_current_discrete = np.zeros(Ci.shape)


    x_recon_lam = 0.001
    x_recon_R = 1.0
    delta_F = 0.1
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

    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')

    bar = ChargingBar('Learning', max=TotTime-1)
    for i in range(2, TotTime):


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
            """Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), utils.Ntime)).T
            for d in range(utils.Nx):
                Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')"""

            # 2)
            X = np.zeros(X.shape)
            for t in range(1, utils.Ntime):
                X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:, t]
            # 3)
            sbs.load_signal(np.copy(X))
            # 4)
            for i in range(delta_Omega.shape[1]): # First transform each column
                if(ks[i] > 0):
                    delta_Omega[:,i] /= ks[i]
            C_current_discrete = sbs.set_omega_stochastic_round(C_real=np.copy(C), delta_C_real=np.copy(delta_Omega), stochastic=False)
            # 5)
            sbs.set_recurrent_connection()
            # Do the update
            C = C - np.copy(delta_Omega)
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
                V_recons[:,t] = np.copy(new_V_recon) #! Check if copy necessary
            
        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = np.copy(C) # Indexing starts at 0
            Cs_discrete[j-1,:,:] = np.copy(C_current_discrete)
            j = j+1
        bar.next()
    bar.next()
    bar.finish()

    Cs_discrete[utils.T-1,:,:] = np.copy(C_current_discrete)
    Cs[utils.T-1,:,:] = np.copy(C)

    ########## Compute the optimal decoder ##########
    TimeL = 5000 #! Change to 50000, really? Would take 8 min for one decoder.
    xL = np.zeros((utils.Nx, TimeL))
    Decs = np.zeros([utils.T, utils.Nx, utils.Nneuron])

    # Generate new input
    InputL = 0.3*utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeL)).T
    for d in range(utils.Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')

    # Compute the target output by a leaky integration of the input
    for t in range(1,TimeL):
        xL[:,t] = (1-utils.lam*utils.dt)*xL[:,t-1] + utils.dt*InputL[:,t-1]

    # Load the new input into the spike generator
    print("Loading new signal...")
    sbs.load_signal(xL)

    print(("Computing %d decoders" % utils.T))

    bar = ChargingBar('Decoders', max=utils.T)
    for i in range(utils.T):
        (rOL,_,_) = runnet(sbs, utils, F, Cs_discrete[i,:,:], TimeL, xL, x_recon_lam = x_recon_lam, x_recon_R = x_recon_R, delta_F=delta_F)
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

    Trials = 2
    for r in range(Trials):
        InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T

        for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

        # Compute the target output by leaky integration of InputT
        for t in range(1,TimeT):
            xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]
        
        # Load the input
        sbs.load_signal(np.copy(xT))

        bar = ChargingBar(('Error #%d' % r), max=utils.T)
        for i in range(utils.T):
            (rOT, OT, VT) = runnet(sbs, utils, F, Cs_discrete[i,:,:], TimeT, xT, x_recon_lam = x_recon_lam, x_recon_R = x_recon_R, delta_F=delta_F)
            xestc = np.matmul(Decs[i,:,:], rOT) # Decode the rate vector
            Error[0,i] = Error[0,i] + np.sum(np.var(xT-xestc, axis=1, ddof=1)) / (np.sum(np.var(xT, axis=1, ddof=1))*Trials)
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