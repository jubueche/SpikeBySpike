import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from runnet import runnet


# TODO Need to save figures in direc
def plot(results, utils, direc):

    Error = results["Error"]
    MeanPrate = results["MeanPrate"]
    MembraneVar = results["MembraneVar"]
    ErrorC = results["ErrorC"]
    T = utils.T
    dt = utils.dt

    loglog_x = 2**np.linspace(1,T,T)

    plt.figure(figsize=(12, 6))

    plt.subplot(311)
    plt.plot(loglog_x*dt, Error.reshape((-1,1)), 'k')
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time')
    plt.ylabel('Decoding Error')
    plt.title('Evolution of the Decoding Error Through Learning')

    plt.subplot(312)
    plt.plot(loglog_x*dt, MeanPrate.reshape((-1,1)), 'k')
    plt.xscale('log')
    plt.xlabel('Time')
    plt.ylabel('Mean Rate per neuron')
    plt.title('Evolution of the Mean Population Firing Rate Through Learning')

    plt.subplot(313)
    plt.plot(loglog_x*dt, MembraneVar.reshape((-1,1)), 'k')
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time')
    plt.ylabel('Voltage Variance per Neuron')
    plt.title('Evolution of the Variance of the Membrane Potential')

    plt.tight_layout()
    plt.savefig(os.path.join(direc, "errors.png"))

    plt.figure(figsize=(12, 6))

    plt.plot(loglog_x*dt, ErrorC.reshape((-1,1)), 'k')
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time')
    plt.ylabel('Distance to optimal weights')
    plt.title('Weight Convergence')

    plt.tight_layout()
    plt.savefig(os.path.join(direc, "weight_convergence.png"))
    plt.show()

def plot_from_resources(resources_direc, utils, direc):

    # Generate new input signal
    # Load pre weights
    Fi = np.load(os.path.join(resources_direc, "Fi.dat"), allow_pickle=True)
    Ci = np.load(os.path.join(resources_direc, "Ci.dat"), allow_pickle=True)
    Deci = np.load(os.path.join(resources_direc, "Deci.dat"), allow_pickle=True)
    # Load post weights
    F_after = np.load(os.path.join(resources_direc, "F_after.dat"), allow_pickle=True)
    C_after = np.load(os.path.join(resources_direc, "C_after.dat"), allow_pickle=True)
    Dec_after = np.load(os.path.join(resources_direc, "D_after.dat"), allow_pickle=True)
    # Load kernel
    w = np.load(os.path.join(resources_direc, "w.dat"), allow_pickle=True)
    # Generate new test input
    TimeT = 1000
    xT = np.zeros((utils.Nx, TimeT))
    InputT = utils.A*(np.random.multivariate_normal(np.zeros(utils.Nx), np.eye(utils.Nx), TimeT)).T
    for d in range(utils.Nx):
            InputT[d,:] = np.convolve(InputT[d,:], w, 'same')

    # Compute the target output by leaky integration of InputT
    for t in range(1,TimeT):
        xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

    # Run on beginning
    (rOT_initial, OT_initial, VT_initial) = runnet(utils.dt, utils.lam, Fi, InputT, Ci, utils.Nneuron, TimeT, utils.Thresh)
    xest_initial = np.matmul(Deci, rOT_initial)

    # Run on end
    (rOT_after, OT_after, VT_after) = runnet(utils.dt, utils.lam, F_after, InputT, C_after, utils.Nneuron, TimeT, utils.Thresh)
    xest_after = np.matmul(Dec_after, rOT_after)

    ######### Plotting #########
    plt.figure(figsize=(16, 10))
    subplot = 611
    plt.title('Initial reconstruction (green) of the target signal (red)')
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Initial reconstruction (green) of the target signal (red)')
        plt.plot(xT[i,:], 'r')
        plt.plot(xest_initial[i,:], 'g')
        subplot = subplot+1
    
    # Plot initial spike trains
    plt.subplot(subplot)
    plt.title('Initial spike trains')
    coordinates_intial = np.nonzero(OT_initial)
    plt.scatter(coordinates_intial[1], coordinates_intial[0], s=0.8, marker="o", c="k")
    subplot = subplot+1

    # Plot after learning
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Post-learning reconstruction (green) of the target signal (red)')
        plt.plot(xT[i,:], 'r')
        plt.plot(xest_after[i,:], 'g')
        subplot = subplot+1

    # Plot post-learning spike trains
    plt.subplot(subplot)
    plt.title('Post-learning spike trains')
    coordinates_after = np.nonzero(OT_after)
    plt.scatter(coordinates_after[1], coordinates_after[0], s=0.8, marker="o", c="k")
    subplot = subplot+1

    plt.tight_layout()
    plt.savefig(os.path.join(direc, "after_training.png"))
    plt.show()
