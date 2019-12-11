import numpy as np
from progress.bar import ChargingBar
from Utils import my_max
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
import matplotlib.pyplot as plt
from Utils import Utils

"""
TODO: Check if thresholds change constantly
"""

def get_input(duration, utils, w):
    Input = (np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), duration)).T
    for d in range(utils.Nx):
        Input[d,:] = utils.A*np.convolve(Input[d,:], w, 'same')
    return Input

def runnet(F,C,Input,utils,thresh):
    duration = Input.shape[1] # Assuming X has shape (Nx,time)
    V = np.zeros((utils.Nneuron,duration))
    R = np.zeros((utils.Nneuron,duration))
    OT = np.zeros((utils.Nneuron,duration))

    for t in range(1,duration):
        V[:,t] = ((1-utils.lam*utils.dt)*V[:,t-1].reshape((-1,1)) + utils.dt*np.matmul(F, Input[:,t].reshape((-1,1))) + np.matmul(C,OT[:,t-1]).reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)).ravel()
        (m, k) = my_max(V[:,t].reshape((-1,1)) - thresh.reshape((-1,1))) # Returns maximum and argmax
        if(m>=0):
            OT[k,t] = 1.0
        R[:,t] = (1-utils.lam*utils.dt)*R[:,t-1] + OT[:,t]

    return R,OT,V

def update_firing_time_window(spike_time_window, new_spikes):
    window_size = spike_time_window.shape[1]
    spike_time_window[:,0:-1] = spike_time_window[:,1:]
    spike_time_window[:,-1] = new_spikes.ravel()
    return spike_time_window

def get_recon_error(F,C,thresh,w):
    Input = get_input(utils.Ntime,utils,w)
    X = np.zeros((utils.Nx,utils.Ntime))
    # Compute the target output by leaky integration of InputT
    for t in range(1,utils.Ntime):
        X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:,t-1]

    R,_,_ = runnet(F,C,Input,utils,thresh)
    Dec = np.linalg.lstsq(R.T, X.T, rcond=None)[0].T

    Input = get_input(utils.Ntime,utils,w)
    X = np.zeros((utils.Nx,utils.Ntime))
    # Compute the target output by leaky integration of InputT
    for t in range(1,utils.Ntime):
        X[:,t] = (1-utils.lam*utils.dt)*X[:,t-1] + utils.dt*Input[:,t-1]

    R,OT,V = runnet(F,C,Input,utils,thresh)
    xestc = np.matmul(Dec, R)
    err = np.sum(np.var(X-xestc, axis=1, ddof=1)) / (np.sum(np.var(X, axis=1, ddof=1)))
    return OT,err,xestc,X


def Learning(utils, F, C):
    TotTime = utils.Nit*utils.Ntime
    Fi = np.copy(F)
    Ci = np.copy(C)
    thresh = np.ones(utils.Nneuron)*utils.Thresh

    Cs = np.zeros([utils.T, utils.Nneuron, utils.Nneuron]) # Store the recurrent weights over the course of training
    Copt = np.matmul(-F, F.T)

    V = np.zeros((utils.Nneuron, 1))
    Vs = np.zeros((utils.Nneuron, utils.Ntime))

    spike_time_window_size = 10
    spike_time_window = np.zeros((utils.Nneuron,spike_time_window_size))
    count = 0

    O = 0
    k = 0 #! Indexing starts with 0
    r0 = np.zeros((utils.Nneuron, 1))
    R = np.zeros((utils.Nneuron,utils.Ntime))
    ot = np.zeros((utils.Nneuron, 1))
    OT = np.zeros((utils.Nneuron, utils.Ntime))
    distance_to_optimal_weights = []
    mean_firing_rate = []
    Error = []
    number_updates = []

    x = np.zeros((utils.Nx, 1))
    X = np.zeros((utils.Nx, utils.Ntime))

    Input = np.zeros((utils.Nx, utils.Ntime))
    Id = np.eye(utils.Nneuron)

    w = (1/(utils.sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,1000)-500)**2)/(2*utils.sigma**2))
    w = w / np.sum(w)

    ###### Tracking connection k -> n
    V_n_before = []
    V_n_after = []
    spike_times_n_k = []
    correction_times = []
    upper_bound = []
    lower_bound = []
    lower_bound_global = []
    upper_bound_global = []
    distances_to_optimal_weights = []
    # Random neurons
    neuron_k = 48
    neuron_n = 15
    iteration_to_record = [1]
    
    j = 1
    number_of_signal_iterations = 0

    bar = ChargingBar('Learning', max=TotTime-1)
    for i in range(2, TotTime):

        if((i % 2**j) == 0): # Save the matrices on an exponential scale
            Cs[j-1,:,:] = C # Indexing starts at 0
            j = j+1

        if(((i-2) % utils.Ntime) == 0):
            # Plotting if number of signal iterations is 1
            if(number_of_signal_iterations == iteration_to_record[-1]):
                plt.figure(figsize=(14,5))
                plt.title('Behaviour of the learning algorithm under optimal weights')
                mean_diff_size = np.mean(np.asarray(V_n_after) - np.asarray(V_n_before))
                mean_diff_around = np.mean(np.asarray(V_n_after) + np.asarray(V_n_before))
                for idx,t in enumerate(spike_times_n_k):
                    if(t in correction_times):
                        plt.plot((t,t), (V_n_after[idx],V_n_before[idx]), '-', color='r')
                    else:
                        plt.plot((t,t), (V_n_after[idx],V_n_before[idx]), '-', color='y')
                plt.plot(spike_times_n_k,V_n_before, 'bo', markersize=4.0, label=r"$V_{\textnormal{before}}$")
                plt.plot(spike_times_n_k,V_n_after, 'go', markersize=4.0, label=r"$V_{\textnormal{after}}$")
                plt.plot(upper_bound, label="Upper bound")
                plt.plot(lower_bound, label="Lower bound")
                plt.ylabel(r"V(t)")
                plt.xlabel(r"t (ms)")
                plt.axhline(y=mean_diff_size, color=(0,1,1,1.0), label=r"Mean $V_{\textnormal{before}} - V_{\textnormal{after}}$")
                plt.axhline(y=mean_diff_around, color=(0.4,0.4,0.1,1.0), label=r"Mean $V_{\textnormal{before}} + V_{\textnormal{after}}$")

                plt.axhline(y=utils.Thresh, linestyle='--', color=(0,0.2,0.4,1.0), label=r"Optimal $V_{\textnormal{before}} - V_{\textnormal{after}}$")
                plt.axhline(y=0, linestyle='--', color=(0.8,0.2,0.1,1.0), label=r"Optimal $V_{\textnormal{before}} + V_{\textnormal{after}}$")
                
                plt.xlim([0,1000*len(iteration_to_record)])
                plt.legend(bbox_to_anchor=(1,0.75), loc="upper left")
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(12,4))
                plt.plot(distances_to_optimal_weights)
                plt.show()

            Input = get_input(utils.Ntime, utils, w)
            V *= 0
            spike_time_window *= 0
            count = 0
            number_of_updates = 0
            number_of_signal_iterations += 1
            

        t = ((i-2) % utils.Ntime)

        V = (1-utils.lam*utils.dt)*V + utils.dt*np.matmul(F, Input[:,t].reshape((-1,1))) + np.matmul(C,ot).reshape((-1,1)) + 0.001*np.random.randn(utils.Nneuron, 1)
        Vs[:,t] = V.ravel()

        x = (1-utils.lam*utils.dt)*x + utils.dt*Input[:,t].reshape((-1,1))
        X[:,t] = x.ravel()

        (m, k) = my_max(V - thresh.reshape((-1,1))-0.001*np.random.randn(utils.Nneuron, 1)) # Returns maximum and argmax
        ot = np.zeros((utils.Nneuron,1))
        if(m>=0):
            ot[k] = 1.0
            if(k == neuron_k and number_of_signal_iterations in iteration_to_record):
                spike_times_n_k.append((number_of_signal_iterations-1)*1000 + t)

        OT[:,t] = ot.ravel()
        V_after = V + np.matmul(C,ot).reshape((-1,1))

        if(m>=0 and neuron_k == k and number_of_signal_iterations in iteration_to_record):
            V_n_after.append(V_after[neuron_n])
            V_n_before.append(V[neuron_n])

        margin = 10
        # Do the update here! If neuron k spiked. Also do it using a V_after average.
        if(m>=0): # Also check if the mean is representative
            for n in range(utils.Nneuron):
                if(not n == k):
                    if(2*V_after[n] - C[n,k] <  -1/2 - margin):
                        C[n,k] += 1
                        number_of_updates += 1
                        if(k == neuron_k and n == neuron_n and number_of_signal_iterations in iteration_to_record):
                            correction_times.append((number_of_signal_iterations-1)*1000 + t)
                    elif(2*V_after[n] - C[n,k] > 1/2 + margin):
                        C[n,k] -= 1
                        number_of_updates += 1
                        if(k == neuron_k and n == neuron_n and number_of_signal_iterations in iteration_to_record):
                            correction_times.append((number_of_signal_iterations-1)*1000 + t)

        if(number_of_signal_iterations in iteration_to_record):
            lower_bound.append((-1/2-margin+C[neuron_n,neuron_k])/2)
            upper_bound.append((1/2+margin+C[neuron_n,neuron_k])/2)

        lower_bound_global.append((-1/2-margin+C[neuron_n,neuron_k])/2)
        upper_bound_global.append((1/2+margin+C[neuron_n,neuron_k])/2)

        if(number_of_signal_iterations in iteration_to_record):
            optscale = np.trace(np.matmul(C.T, Copt)) / np.sum(Copt**2)
            Cnorm = np.sum(C**2)
            distances_to_optimal_weights.append(np.sum(np.sum((C - optscale*Copt)**2 ,axis=0)) / Cnorm)

        r0 = (1-utils.lam*utils.dt)*r0 + ot
        R[:,t] = r0.ravel()

        # Update spike time window
        spike_time_window = update_firing_time_window(spike_time_window,ot)
        firing_rates = np.mean(spike_time_window, axis=1)

        if(count < spike_time_window_size):
            count += 1

        if(count == spike_time_window_size):
            too_high_freq = np.linspace(0,utils.Nneuron-1,utils.Nneuron)[firing_rates > 0.6].astype(np.int)
            thresh[too_high_freq] += 0.1


        if((i-2) % (utils.Ntime-1) == 0):
            Copt = np.matmul(-F, F.T)
            optscale = np.trace(np.matmul(C.T, Copt)) / np.sum(Copt**2)
            Cnorm = np.sum(C**2)
            ErrorC = np.sum(np.sum((C - optscale*Copt)**2 ,axis=0)) / Cnorm
            OT,decoding_error,_,_ = get_recon_error(F,C,thresh,w)
            ###
            """coordinates = np.nonzero(OT)
            plt.scatter(coordinates[1],coordinates[0], marker='o', s=0.1)
            plt.show()"""
            ###
            Error.append(decoding_error)
            MeanFrate = np.sum(OT) / (utils.Ntime*utils.dt*utils.Nneuron)
            print("Distance to optimal weights: %.3f Number of updates: %d Decoding error: %.3f Mean F-rate: %.3f" % (ErrorC,number_of_updates,decoding_error,MeanFrate))
            distance_to_optimal_weights.append(ErrorC)
            mean_firing_rate.append(MeanFrate)
            if(not i == 2):
                number_updates.append(number_of_updates)

        bar.next()
    bar.next(); bar.finish()

    axis_font_size = 5

    plt.figure(figsize=(14,3))
    plt.title("Evolution of the bounds under optimal weights")
    plt.plot(upper_bound_global, label="Upper bound")
    plt.plot(lower_bound_global, label="Lower bound")
    plt.xlabel("t (ms)")
    plt.ylabel(r"Bound: $(\pm (\frac{1}{2} + \mu)+\Omega_{n,k})/2$")
    plt.legend(bbox_to_anchor=(1,0.75), loc="upper left")
    plt.tight_layout()
    plt.show()

    OT,err,xestc,X = get_recon_error(F,C,thresh,w)
    plt.figure(figsize=(15,6))
    plt.subplot(611)
    plt.plot(X.T,label="True")
    plt.plot(xestc.T, label="Reconstructed")
    plt.ylabel("Reconstructed signal", fontsize=axis_font_size)
    plt.legend()

    plt.subplot(612)
    coordinates = np.nonzero(OT)
    plt.scatter(coordinates[1],coordinates[0], marker='o', s=0.1)
    plt.xlim([0,utils.Ntime])
    plt.ylim([0,utils.Nneuron])
    plt.ylabel("Neuron ID", fontsize=axis_font_size)

    plt.subplot(613)
    plt.plot(distance_to_optimal_weights)
    plt.ylim([0,1])
    plt.ylabel("Distance to optimal weights", fontsize=axis_font_size)
    
    plt.subplot(614)
    plt.plot(Error)
    plt.ylabel("Decoding error" ,fontsize=axis_font_size)

    plt.subplot(615)
    plt.plot(mean_firing_rate)
    plt.ylabel("Mean firing rate" ,fontsize=axis_font_size)

    plt.subplot(616)
    plt.plot(number_updates)
    plt.ylabel("Number of weight updates" ,fontsize=axis_font_size)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.matshow(Ci, fignum=False)
    plt.subplot(122)
    plt.matshow(C, fignum=False)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


############# Execute the learning process
np.random.seed(42)

utils = Utils(Nneuron=50, Nx=2, lam=50, dt=0.001, epsr=0.001, epsf=0.0001, alpha=0.18, beta=1.11, mu=0.022,
            gamma=1.0, Thresh=4, Nit=60, Ntime=1000, A=2000, sigma=30,
            dynapse_maximal_synapse_ff=8, dynapse_maximal_synapse_o=2, alignment_delta_t=10)

angles = np.linspace(0,2*np.pi,num=utils.Nneuron+1)[:-1]
D = np.vstack((np.cos(angles),np.sin(angles)))
D = D[:utils.Nx,:]
D = np.round(2.6*D).astype(np.int)
D = np.round(1.8*np.random.randn(utils.Nx,utils.Nneuron)).astype(np.int)
F = D.T
print(F)
# Optimal recurrent connection
# Omega = -np.matmul(F,F.T)

# Random recurrent initial connections
Omega = np.round(0.8*np.random.randn(utils.Nneuron,utils.Nneuron)).astype(np.int)
np.fill_diagonal(Omega,-utils.Thresh)
print(Omega)

Learning(utils,F=F, C=Omega)
