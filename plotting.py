import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from runnet import *


def plot_DYNAPS(utils, direc):

    try:
        Error = np.load("Resources/DYNAPS/DYNAPS_Error.dat", allow_pickle=True)
        MeanPrate = np.load("Resources/DYNAPS/DYNAPS_MeanPrate.dat", allow_pickle=True)
        MembraneVar = np.load("Resources/DYNAPS/DYNAPS_MembraneVar.dat", allow_pickle=True)
        ErrorC = np.load("Resources/DYNAPS/DYNAPS_ErrorC.dat", allow_pickle=True)
        O_DYNAPS_initial = np.load("Resources/DYNAPS/O_DYNAPS_initial.dat", allow_pickle=True)
        O_DYNAPS_after = np.load("Resources/DYNAPS/O_DYNAPS_after.dat", allow_pickle=True)
        xestc_initial = np.load("Resources/DYNAPS/DYNAPS_xestc_initial.dat", allow_pickle=True)
        xestc_after = np.load("Resources/DYNAPS/DYNAPS_xestc_after.dat", allow_pickle=True)
        X = np.load("Resources/DYNAPS/DYNAPS_xT.dat", allow_pickle=True)

    except:
        print("Error loading data.")
        return


    title_font_size = 6
    axis_font_size = 5
    ticks_font_size = 5
    linewidth = 0.5

    color = 'C1'
    color_true = 'C1'
    color_recon = 'C2'
    color_third = 'C3'
    markersize = 0.00001
    marker = ','
    markercolor = 'b'
    alpha = 1.0


    plt.figure(figsize=(6.00, 5.51))
    subplot = 611
    plt.title('Initial reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Initial reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
        plt.plot(X[i,:], color=color_true, linewidth=linewidth)
        plt.plot(xestc_initial[i,:], color=color_recon, linewidth=linewidth)
        plt.xticks([],[]); plt.yticks([],[])
        subplot = subplot+1
    
    # Plot initial spike trains
    plt.subplot(subplot)
    plt.title('Initial spike trains', fontname="Times New Roman" ,fontsize=title_font_size)
    coordinates_intial = np.nonzero(O_DYNAPS_initial)
    plt.scatter(coordinates_intial[1], coordinates_intial[0], s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    subplot = subplot+1

    # Plot after learning
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Post-learning reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
        plt.plot(X[i,:], color=color_true, linewidth=linewidth)
        plt.plot(xestc_after[i,:], color=color_recon, linewidth=linewidth)
        plt.xticks([],[]); plt.yticks([],[])
        subplot = subplot+1

    # Plot post-learning spike trains
    plt.subplot(subplot)
    plt.title('Post-learning spike trains', fontname="Times New Roman" ,fontsize=title_font_size)
    coordinates_after = np.nonzero(O_DYNAPS_after)
    plt.scatter(coordinates_after[1], coordinates_after[0], s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    subplot = subplot+1

    plt.tight_layout()    
    name = "DYNAPS_reconstruction.eps"
    plt.savefig(os.path.join(direc, name), format="eps")
    plt.show()


    #################################################################################

    T = utils.T
    dt = utils.dt

    loglog_x = 2**np.linspace(1,T,T)

    plt.figure(figsize=(6.010, 4.73))

    plt.subplot(411)

    plt.plot(loglog_x, ErrorC.reshape((-1,1)), color=color, linewidth=linewidth)

    plt.xscale('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Distance to optimal weights', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Weight Convergence', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)

    plt.subplot(412)
    
    plt.plot(loglog_x, Error.reshape((-1,1)), color=color, linewidth=linewidth)
    
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Decoding Error', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Decoding Error Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)

    plt.subplot(413)
    plt.plot(loglog_x, MeanPrate.reshape((-1,1)), color=color, linewidth=linewidth)
    plt.xscale('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Mean Rate per neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Mean Population Firing Rate Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)

    plt.subplot(414)
    plt.plot(loglog_x, MembraneVar.reshape((-1,1)), color=color, linewidth=linewidth)
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Voltage Variance per Neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Variance of the Membrane Potential', fontname="Times New Roman" ,fontsize=title_font_size)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)

    plt.tight_layout()
    name = "DYNAPS_convergence.eps"
    plt.savefig(os.path.join(direc, name), format="eps")

    plt.tight_layout()
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

    (OT_down, OT_up) = get_spiking_input(utils.delta_modulator_threshold, InputT, utils.Nx, TimeT)

    # Run on beginning
    (rOT_initial, OT_initial, VT_initial) = runnet_recon_x(utils.dt, utils.lam, Fi, OT_up, OT_down, Ci, utils.Nneuron, TimeT, utils.Thresh, xT, x_recon_lam = 0.001, x_recon_R = 1.0)
    xest_initial = np.matmul(Deci, rOT_initial)

    # Run on end
    (rOT_after, OT_after, VT_after) = runnet_recon_x(utils.dt, utils.lam, F_after, OT_up, OT_down, C_after, utils.Nneuron, TimeT, utils.Thresh, xT, x_recon_lam = 0.001, x_recon_R = 1.0)
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
