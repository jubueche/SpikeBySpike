"""from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from runnet import *
from Learning import *
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter

def plot_DYNAPS(utils, direc):

    try:
        O_DYNAPS_initial = np.load("Resources/DYNAPS/O_DYNAPS_initial.dat", allow_pickle=True)
        O_DYNAPS_after = np.load("Resources/DYNAPS/O_DYNAPS_after.dat", allow_pickle=True)
        xestc_initial = np.load("Resources/DYNAPS/DYNAPS_xestc_initial.dat", allow_pickle=True)
        xestc_after = np.load("Resources/DYNAPS/DYNAPS_xestc_after.dat", allow_pickle=True)
        X = np.load("Resources/DYNAPS/DYNAPS_xT.dat", allow_pickle=True)

    except:
        print("Error loading data.")
        return

    try:
        Error5 = np.load("Resources/DYNAPS/DYNAPS_Error_5.dat", allow_pickle=True)
        MeanPrate5 = np.load("Resources/DYNAPS/DYNAPS_MeanPrate_5.dat", allow_pickle=True)
        MembraneVar5 = np.load("Resources/DYNAPS/DYNAPS_MembraneVar_5.dat", allow_pickle=True)
        ErrorC5 = np.load("Resources/DYNAPS/DYNAPS_ErrorC_5.dat", allow_pickle=True)

        Error10 = np.load("Resources/DYNAPS/DYNAPS_Error_10.dat", allow_pickle=True)
        MeanPrate10 = np.load("Resources/DYNAPS/DYNAPS_MeanPrate_10.dat", allow_pickle=True)
        MembraneVar10 = np.load("Resources/DYNAPS/DYNAPS_MembraneVar_10.dat", allow_pickle=True)
        ErrorC10 = np.load("Resources/DYNAPS/DYNAPS_ErrorC_10.dat", allow_pickle=True)

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


    plt.figure(figsize=(6.00, 3.94))
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

    plt.figure(figsize=(6.010, 3.94))

    plt.subplot(411)
    ax = plt.gca()

    l1 = ax.plot(loglog_x, ErrorC5.reshape((-1,1)), color=color_true, linewidth=linewidth, label="Time-window size 5ms")
    ax2 = ax.twinx()
    l2 = ax2.plot(loglog_x, ErrorC10.reshape((-1,1)), color=color_recon, linewidth=linewidth, label="Time-window size 10ms")
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    L = ax.legend(lns, labs, loc=0, frameon=False, ncol=1, labelspacing=0.1)
    plt.setp(L.texts, family='Times New Roman',fontsize=5)

    plt.xscale('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Distance to optimal weights', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Weight Convergence', fontname="Times New Roman" ,fontsize=title_font_size)
    ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
    ax.tick_params(axis='x', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)

    plt.subplot(412)
    ax = plt.gca()
    ax.plot(loglog_x, Error5.reshape((-1,1)), color=color_true, linewidth=linewidth)
    ax2 = ax.twinx()
    ax2.plot(loglog_x, Error10.reshape((-1,1)), color=color_recon, linewidth=linewidth)

    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Decoding Error', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Decoding Error Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
    ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
    ax.tick_params(axis='x', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)

    plt.subplot(413)
    ax = plt.gca()
    ax.plot(loglog_x, MeanPrate5.reshape((-1,1)), color=color_true, linewidth=linewidth)
    ax2 = ax.twinx()
    ax2.plot(loglog_x, MeanPrate10.reshape((-1,1)), color=color_recon, linewidth=linewidth)


    plt.xscale('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Mean Rate per neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Mean Population Firing Rate Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
    ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
    ax.tick_params(axis='x', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)

    plt.subplot(414)
    ax = plt.gca()
    ax.plot(loglog_x, MembraneVar5.reshape((-1,1)), color=color_true, linewidth=linewidth)
    ax2 = ax.twinx()
    ax2.plot(loglog_x, MembraneVar10.reshape((-1,1)), color=color_recon, linewidth=linewidth)
    plt.xscale('log')
    plt.ylabel('log')
    plt.xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.ylabel('Voltage Variance per Neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.title('Evolution of the Variance of the Membrane Potential', fontname="Times New Roman" ,fontsize=title_font_size)
    ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
    ax.tick_params(axis='x', labelsize=ticks_font_size)
    ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)

    plt.tight_layout()
    name = "DYNAPS_convergence.eps"
    plt.savefig(os.path.join(direc, name), format="eps")

    plt.tight_layout()
    plt.show()

def plot_from_resources(resources_direc, utils, direc, update_all = False, discretize = False,
                            remove_positive = False, use_spiking=False, use_batched=False,use_batched_nn=False,
                            use_audio=False, audio_helper = None):

    if(use_audio and audio_helper is None):
        raise Exception("Audio helper is None")

    # Generate new input signal
    if(discretize):
        ending = "_d.dat"
    elif(update_all):
        ending = "_ua.dat"
    elif(remove_positive):
        ending = "_rm.dat"
    elif(use_spiking):
        ending = "_us.dat"
    elif(use_batched):
        ending = "_ub.dat"
    elif(use_audio):
        ending = "_audio.dat"
    else:
        ending = ".dat"
    
    try:
        # Load pre weights
        Fi = np.load(os.path.join(resources_direc, ("Fi%s" % ending)), allow_pickle=True)
        Ci = np.load(os.path.join(resources_direc, ("Ci%s" % ending)), allow_pickle=True)
        Deci = np.load(os.path.join(resources_direc, ("Deci%s" % ending)), allow_pickle=True)
        # Load post weights
        F_after = np.load(os.path.join(resources_direc, ("F_after%s" % ending)), allow_pickle=True)
        C_after = np.load(os.path.join(resources_direc, ("C_after%s" % ending)), allow_pickle=True)
        Dec_after = np.load(os.path.join(resources_direc, ("D_after%s" % ending)), allow_pickle=True)

    except:
        print("No data found...")
        return

    have_normal = True
    have_discrete = True
    have_rm = True
    have_spiking = True
    have_ua = True
    have_ub = True
    have_ub_nn = True
    have_audio = True

    title_font_size = 6
    axis_font_size = 5
    ticks_font_size = 5
    linewidth = 0.5

    color = 'C1'
    color_true = 'C1'
    color_recon = 'C2'
    color_third = 'C3'
    markersize = 0.1
    marker = 'o'
    markercolor = 'b'
    alpha = 1.0


    try:
        MembraneVar = np.load(os.path.join(resources_direc, "MembraneVar.dat"), allow_pickle=True)
        MeanPrate = np.load(os.path.join(resources_direc, "MeanPrate.dat"), allow_pickle=True)
        ErrorC = np.load(os.path.join(resources_direc, "ErrorC.dat"), allow_pickle=True)
        Error = np.load(os.path.join(resources_direc, "Error.dat"), allow_pickle=True)
    except:
        print("No normal data")
        have_normal = False

    try:
        MembraneVar_ua = np.load(os.path.join(resources_direc, "MembraneVar_ua.dat"), allow_pickle=True)
        MeanPrate_ua = np.load(os.path.join(resources_direc, "MeanPrate_ua.dat"), allow_pickle=True)
        ErrorC_ua = np.load(os.path.join(resources_direc, "ErrorC_ua.dat"), allow_pickle=True)
        Error_ua = np.load(os.path.join(resources_direc, "Error_ua.dat"), allow_pickle=True)
    except:
        print("No update using all neurons data")
        have_ua = False

    try:
        C_after_disc = np.load(os.path.join(resources_direc, "C_after_d.dat"), allow_pickle=True)
        MembraneVar_d = np.load(os.path.join(resources_direc, "MembraneVar_d.dat"), allow_pickle=True)
        MeanPrate_d = np.load(os.path.join(resources_direc, "MeanPrate_d.dat"), allow_pickle=True)
        ErrorC_d = np.load(os.path.join(resources_direc, "ErrorC_d.dat"), allow_pickle=True)
        Error_d = np.load(os.path.join(resources_direc, "Error_d.dat"), allow_pickle=True)
    except:
        print("No discretized data found")
        have_discrete = False

    try:
        MembraneVar_rm = np.load(os.path.join(resources_direc, "MembraneVar_rm.dat"), allow_pickle=True)
        MeanPrate_rm = np.load(os.path.join(resources_direc, "MeanPrate_rm.dat"), allow_pickle=True)
        ErrorC_rm = np.load(os.path.join(resources_direc, "ErrorC_rm.dat"), allow_pickle=True)
        Error_rm = np.load(os.path.join(resources_direc, "Error_rm.dat"), allow_pickle=True)
    except:
        print("No remove positive weights data")
        have_rm = False

    try:
        MembraneVar_us = np.load(os.path.join(resources_direc, "MembraneVar_us.dat"), allow_pickle=True)
        MeanPrate_us = np.load(os.path.join(resources_direc, "MeanPrate_us.dat"), allow_pickle=True)
        ErrorC_us = np.load(os.path.join(resources_direc, "ErrorC_us.dat"), allow_pickle=True)
        Error_us = np.load(os.path.join(resources_direc, "Error_us.dat"), allow_pickle=True)
    except:
        print("No use spiking input data")
        have_spiking = False

    try:
        MembraneVar_ub = np.load(os.path.join(resources_direc, "MembraneVar_ub.dat"), allow_pickle=True)
        MeanPrate_ub = np.load(os.path.join(resources_direc, "MeanPrate_ub.dat"), allow_pickle=True)
        ErrorC_ub = np.load(os.path.join(resources_direc, "ErrorC_ub.dat"), allow_pickle=True)
        Error_ub = np.load(os.path.join(resources_direc, "Error_ub.dat"), allow_pickle=True)
    except:
        print("No use batched update data")
        have_ub = False

    try:
        MembraneVar_ub_nn = np.load(os.path.join(resources_direc, "MembraneVar_ub_nn.dat"), allow_pickle=True)
        MeanPrate_ub_nn = np.load(os.path.join(resources_direc, "MeanPrate_ub_nn.dat"), allow_pickle=True)
        ErrorC_ub_nn = np.load(os.path.join(resources_direc, "ErrorC_ub_nn.dat"), allow_pickle=True)
        Error_ub_nn = np.load(os.path.join(resources_direc, "Error_ub_nn.dat"), allow_pickle=True)
    except:
        print("No use batched update not normalized data")
        have_ub_nn = False

    try:
        MembraneVar_audio = np.load(os.path.join(resources_direc, "MembraneVar_audio.dat"), allow_pickle=True)
        MeanPrate_audio = np.load(os.path.join(resources_direc, "MeanPrate_audio.dat"), allow_pickle=True)
        ErrorC_audio = np.load(os.path.join(resources_direc, "ErrorC_audio.dat"), allow_pickle=True)
        Error_audio = np.load(os.path.join(resources_direc, "Error_audio.dat"), allow_pickle=True)
    except:
        print("No use batched update not normalized data")
        have_audio = False
   

    # Load kernel
    w = np.load(os.path.join(resources_direc, ("w%s" % ending)), allow_pickle=True)
    # Generate new test input
    if(use_audio):
        TimeT = 500
    else:
        TimeT = 1000
    xT = np.zeros((utils.Nx, TimeT))

    if(use_audio):
        label, InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False, digit=7)
    else:
        InputT = get_input(TimeT, utils, w, audio_helper=audio_helper, use_audio=use_audio,training=False)

    # Compute the target output by leaky integration of InputT
    for t in range(1,TimeT):
        xT[:,t] = (1-utils.lam*utils.dt)*xT[:,t-1] + utils.dt*InputT[:,t-1]

    #### Plot for showing a delta modulated input
    """(OT_up, OT_down) = get_spiking_input(0.5, xT, utils.Nx, TimeT)
    up = OT_up[0,:]
    down = OT_down[0,:]
    coordinates_up = np.nonzero(up)
    coordinates_down = np.nonzero(down)

    plt.figure(figsize=(6.00,2.3))
    plt.subplot(311)
    plt.plot(xT[0,:],color=color_true, linewidth=linewidth)
    plt.xticks([],[]); plt.yticks([],[])
    plt.xlim([0,1000])
    plt.subplot(312)
    plt.scatter(coordinates_up[0],np.zeros(len(coordinates_up[0])),marker = marker, c=markercolor, s=markersize)
    plt.xticks([],[]); plt.yticks([],[])
    plt.xlim([0,1000])
    plt.subplot(313)
    plt.scatter(coordinates_down[0],np.zeros(len(coordinates_down[0])),marker = marker, c=markercolor, s=markersize)
    plt.xlim([0,1000])
    plt.xticks([],[]); plt.yticks([],[])
    plt.savefig(os.path.join(direc, "delta_modulated_input.eps"), format="eps")
    plt.show()"""

    # Run on beginning
    (rOT_initial, OT_initial, VT_initial) = runnet(utils,utils.dt, utils.lam, Fi, InputT, Ci, utils.Nneuron, TimeT, utils.Thresh,use_spiking=use_spiking)
    xest_initial = np.matmul(Deci, rOT_initial)

    # Run on end
    (rOT_after, OT_after, VT_after) = runnet(utils,utils.dt, utils.lam, F_after, InputT, C_after, utils.Nneuron, TimeT, utils.Thresh,use_spiking=use_spiking)
    xest_after = np.matmul(Dec_after, rOT_after)

    if(use_audio):

        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        order = 6
        fs = 1000
        cutoff = 100
        b, a = butter_highpass(cutoff, fs, order)

        # Rescale and write to .wav file
        to_write = 2*xest_after.T / (max(xest_after.T) - min(xest_after.T))
        to_write += np.mean(to_write)
        
        #to_write = 2*(InputT.T  + np.mean(InputT.T))/ (max(InputT.T) - min(InputT.T))
        #to_write =  ((stereoAudio * bits16max)).astype('int16')
        to_write_filtered = butter_lowpass_filter(to_write, cutoff, fs, order)

        plt.plot(to_write)
        plt.plot(to_write_filtered)
        plt.show()

        print(label)
        to_write = np.reshape(to_write, (-1,1))
        write(os.path.join(os.getcwd(), "DYNAPS/Resources/Simulation/recon_audio.wav"), rate = 1000, data=to_write.astype(np.float32))
    

    if(have_discrete):
        (OT_after_disc,_,_) = runnet(utils,utils.dt, utils.lam, F_after, InputT, C_after_disc, utils.Nneuron, TimeT, utils.Thresh,use_spiking=use_spiking)




    ############################################################################
    # Compare the spike trains after training using discretized and continous weights

    """plt.figure(figsize=(6.00,1.57))
    c_after = np.nonzero(OT_after)
    c_after_d = np.nonzero(OT_after_disc)
    plt.subplot(211)
    plt.title('Spike train after training using continous matrix', fontname="Times New Roman" ,fontsize=title_font_size)
    plt.scatter(c_after[1], c_after[0]+1, s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    plt.subplot(212)
    plt.title('Spike train after training using discrete matrix', fontname="Times New Roman" ,fontsize=title_font_size)
    plt.scatter(c_after_d[1], c_after_d[0]+1, s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    plt.tight_layout()
    plt.savefig(os.path.join(direc, "spike_train_cont_vs_disc.eps"), format="eps")
    plt.show()"""


    ##########################################################################
    # Generate classic 2D input

    """plt.figure(figsize=(6.00, 1.18))
    plt.title('Uncorrelated two-component continous signal', fontname="Times New Roman" ,fontsize=title_font_size)
    plt.plot(InputT[0,:], linewidth = linewidth, color="C1")
    plt.plot(InputT[1,:], linewidth = linewidth, color="C2")
    plt.yticks([],[])
    plt.xlabel("Time [ms]",fontname="Times New Roman" ,fontsize=axis_font_size)
    plt.tight_layout()
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_font_size)
    plt.savefig(os.path.join(direc, "input.eps"), format="eps")
    plt.show()"""

    ##########################################################################
    # Generate reconstruction with spike trains pre- and post-learning

    plt.figure(figsize=(6.00, 5.51))
    subplot = 611
    plt.title('Initial reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Initial reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
        plt.plot(xT[i,:], color=color_true, linewidth=linewidth)
        plt.plot(xest_initial[i,:], color=color_recon, linewidth=linewidth)
        plt.xticks([],[]); plt.yticks([],[])
        subplot = subplot+1
    
    # Plot initial spike trains
    plt.subplot(subplot)
    plt.title('Initial spike trains', fontname="Times New Roman" ,fontsize=title_font_size)
    coordinates_intial = np.nonzero(OT_initial)
    plt.scatter(coordinates_intial[1], coordinates_intial[0], s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    subplot = subplot+1

    # Plot after learning
    for i in range(utils.Nx):
        plt.subplot(subplot)
        if(i==0):
            plt.title('Post-learning reconstruction (green) of the target signal (red)', fontname="Times New Roman" ,fontsize=title_font_size)
        plt.plot(xT[i,:], color=color_true, linewidth=linewidth)
        plt.plot(xest_after[i,:], color=color_recon, linewidth=linewidth)
        plt.xticks([],[]); plt.yticks([],[])
        subplot = subplot+1

    # Plot post-learning spike trains
    plt.subplot(subplot)
    plt.title('Post-learning spike trains', fontname="Times New Roman" ,fontsize=title_font_size)
    coordinates_after = np.nonzero(OT_after)
    plt.scatter(coordinates_after[1], coordinates_after[0], s=markersize, marker=marker, c=markercolor, alpha=alpha)
    plt.xticks([],[]); plt.yticks([],[])
    subplot = subplot+1

    plt.tight_layout()
    if(discretize):
        name = "reconstruction_d.eps"
    elif(update_all):
        name = "reconstruction_ua.eps"
    elif(remove_positive):
        name = "reconstruction_rm.eps"
    elif(use_spiking):
        name = "reconstruction_us.eps"
    elif(use_batched):
        name = "reconstruction_ub.eps"
    elif(use_audio):
        name = "reconstructed_audio.eps"
    else:
        name = "reconstruction.eps"
    plt.savefig(os.path.join(direc, name), format="eps")
    plt.show()


    ##########################################################################
    # Plot convergence of either discretized, removed positive, update all or normal
    T = utils.T
    dt = utils.dt

    loglog_x = 2**np.linspace(1,T,T)

    plt.figure(figsize=(6.010, 4.73))

    plt.subplot(411)

    if(discretize):
        plt.plot(loglog_x, ErrorC_d.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(update_all):
        plt.plot(loglog_x, ErrorC_ua.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(remove_positive):
        plt.plot(loglog_x, ErrorC_rm.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_spiking):
        plt.plot(loglog_x, ErrorC_us.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_batched):
        plt.plot(loglog_x, ErrorC_ub.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_audio):
        plt.plot(loglog_x, ErrorC_audio.reshape((-1,1)), color=color, linewidth=linewidth)
    else:
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
    if(discretize):
        plt.plot(loglog_x, Error_d.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(update_all):
        plt.plot(loglog_x, Error_ua.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(remove_positive):
        plt.plot(loglog_x, Error_rm.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_spiking):
        plt.plot(loglog_x, Error_us.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_batched):
        plt.plot(loglog_x, Error_ub.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_audio):
        plt.plot(loglog_x, Error_audio.reshape((-1,1)), color=color, linewidth=linewidth)
    else:
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
    if(discretize):
        plt.plot(loglog_x, MeanPrate_d.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(update_all):
        plt.plot(loglog_x, MeanPrate_ua.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(remove_positive):
        plt.plot(loglog_x, MeanPrate_rm.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_spiking):
        plt.plot(loglog_x, MeanPrate_us.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_batched):
        plt.plot(loglog_x, MeanPrate_ub.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_audio):
        plt.plot(loglog_x, MeanPrate_audio.reshape((-1,1)), color=color, linewidth=linewidth)
    else:    
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
    if(discretize):
        plt.plot(loglog_x, MembraneVar_d.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(update_all):
        plt.plot(loglog_x, MembraneVar_ua.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(remove_positive):
        plt.plot(loglog_x, MembraneVar_rm.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_spiking):
        plt.plot(loglog_x, MembraneVar_us.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_batched):
        plt.plot(loglog_x, MembraneVar_ub.reshape((-1,1)), color=color, linewidth=linewidth)
    elif(use_audio):
        plt.plot(loglog_x, MembraneVar_audio.reshape((-1,1)), color=color, linewidth=linewidth)
    else:
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
    if(discretize):
        name = "convergence_d.eps"
    elif(update_all):
        name = "convergence_ua.eps"
    elif(remove_positive):
        name = "convergence_rm.eps"
    elif(use_spiking):
        name = "convergence_us.eps"
    elif(use_batched):
        name = "convergence_ub.eps"
    elif(use_audio):
        name = "convergence_audio.eps"
    else:
        name = "convergence.eps"
    plt.savefig(os.path.join(direc, name), format="eps")

    plt.tight_layout()
    plt.show()

    
    ##########################################################################
    # definition of function for comparing two convergence behaviours

    def compare_covergence(MPR1,MPR2,ERR1,ERR2,VAR1,VAR2,ERRC1,ERRC2,name, name1, name2,include_bu_nn=False):

        plt.figure(figsize=(6.010, 6.29))

        plt.subplot(411)
        ax = plt.gca()

        l1 = ax.plot(loglog_x, ERRC1.reshape((-1,1)), color=color_true, linewidth=linewidth, label=name1)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        l2 = ax2.plot(loglog_x, ERRC2.reshape((-1,1)), color=color_recon, linewidth=linewidth, label=name2)
        if(include_bu_nn):
            l3 = ax3.plot(loglog_x, ErrorC_ub_nn.reshape((-1,1)), color=color_third, linewidth=linewidth, label="Unnormalized batched update")

        if(include_bu_nn):
            lns = l1+l2+l3
        else:
            lns = l1+l2
        labs = [l.get_label() for l in lns]
        L = ax.legend(lns, labs, loc=0)
        plt.setp(L.texts, family='Times New Roman',fontsize=5)

        ax.set_xscale('log')
        ax.set_xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Distance to optimal weights', fontname="Times New Roman" ,fontsize=axis_font_size)
        plt.title('Weight Convergence', fontname="Times New Roman" ,fontsize=title_font_size)
        
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)
        if(include_bu_nn):
            ax3.tick_params(axis='y', labelsize=ticks_font_size, color=color_third,pad=-12,direction='in')
            ax3.set_yticks(ax3.get_yticks()[2:-2])


        plt.subplot(412)
        ax = plt.gca()
        ax.plot(loglog_x, ERR1.reshape((-1,1)), color=color_true, linewidth=linewidth)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax2.plot(loglog_x, ERR2.reshape((-1,1)), color=color_recon, linewidth=linewidth)
        if(include_bu_nn):
            ax3.plot(loglog_x, Error_ub_nn.reshape((-1,1)), color=color_third, linewidth=linewidth)

        ax.set_xscale('log')
        ax.set_xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Decoding Error', fontname="Times New Roman" ,fontsize=axis_font_size)
        plt.title('Evolution of the Decoding Error Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
        
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)
        if(include_bu_nn):
            ax3.tick_params(axis='y', labelsize=ticks_font_size, color=color_third,pad=-15,direction='in')
            ax3.set_yticks(ax3.get_yticks()[2:-2])

        plt.subplot(413)
        ax = plt.gca()
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax.plot(loglog_x, MPR1.reshape((-1,1)), color=color_true, linewidth=linewidth)
        ax2.plot(loglog_x, MPR2.reshape((-1,1)), color=color_recon, linewidth=linewidth)
        if(include_bu_nn):
            ax3.plot(loglog_x, MeanPrate_ub_nn.reshape((-1,1)), color=color_third, linewidth=linewidth)

        ax.set_xscale('log')
        ax.set_xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Mean Rate per neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
        plt.title('Evolution of the Mean Population Firing Rate Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
        
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)
        if(include_bu_nn):
            ax3.tick_params(axis='y', labelsize=ticks_font_size, color=color_third,pad=-10,direction='in')
            ax3.set_yticks(ax3.get_yticks()[2:-2])

        plt.subplot(414)
        ax = plt.gca()
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax.plot(loglog_x, VAR1.reshape((-1,1)), color=color_true, linewidth=linewidth)
        ax2.plot(loglog_x, VAR2.reshape((-1,1)), color=color_recon, linewidth=linewidth)
        if(include_bu_nn):
            ax3.plot(loglog_x, MembraneVar_ub_nn.reshape((-1,1)), color=color_third, linewidth=linewidth)

        ax.set_xscale('log')
        ax.set_ylabel('log')
        ax.set_xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Voltage Variance per Neuron', fontname="Times New Roman" ,fontsize=axis_font_size)
        plt.title('Evolution of the Variance of the Membrane Potential', fontname="Times New Roman" ,fontsize=title_font_size)
        
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)
        if(include_bu_nn):
            ax3.tick_params(axis='y', labelsize=ticks_font_size, color=color_third,pad=-8,direction='in')
            ax3.set_yticks(ax3.get_yticks()[2:-2])

        plt.tight_layout()
        plt.savefig(os.path.join(direc, name), format="eps")

        plt.tight_layout()
        plt.show()

    ##########################################################################
    # Compare learning using spiking input compared to learning using continous input
    if(have_spiking and have_normal):

        plt.figure(figsize=(6.010, 1.57))

        ax = plt.gca()
        l1 = ax.plot(loglog_x, Error.reshape((-1,1)), color=color_true, linewidth=linewidth, label="Normal learning")
        ax2 = ax.twinx()
        l2 = ax2.plot(loglog_x, Error_us.reshape((-1,1)), color=color_recon, linewidth=linewidth,label="Learning with spiking input")
        
        lns = l1+l2
        labs = [l.get_label() for l in lns]
        L = ax.legend(lns, labs, loc=0)
        plt.setp(L.texts, family='Times New Roman',fontsize=5)

        ax.set_xscale('log')
        ax.set_xlabel('Time', fontname="Times New Roman" ,fontsize=axis_font_size)
        ax.set_ylabel('Decoding Error', fontname="Times New Roman" ,fontsize=axis_font_size)
        plt.title('Evolution of the Decoding Error Through Learning', fontname="Times New Roman" ,fontsize=title_font_size)
        
        ax.tick_params(axis='x', labelsize=ticks_font_size)
        ax.tick_params(axis='y', labelsize=ticks_font_size, color=color_true)
        ax2.tick_params(axis='y', labelsize=ticks_font_size, color=color_recon)

        plt.savefig(os.path.join(direc, "error_convergence_use_spiking_vs_normal.eps"), format="eps")
        plt.show()


        # compare_covergence(MeanPrate,MeanPrate_us,Error,Error_us,MembraneVar,MembraneVar_us,ErrorC,ErrorC_us,"convergence_use_spiking_vs_normal.eps","Normal learning","Learning with spiking input")
    # compare convergence between discretized and updated all
    if(have_discrete and have_ua):
        compare_covergence(MeanPrate_d,MeanPrate_ua,Error_d,Error_ua,MembraneVar_d,MembraneVar_ua,ErrorC_d,ErrorC_ua,"convergence_d_vs_ua.eps","Discretized learning","Updates using all neurons")
    # compare convergence between removed zeros and normal
    if(have_normal and have_rm):
        compare_covergence(MeanPrate,MeanPrate_rm,Error,Error_rm,MembraneVar,MembraneVar_rm,ErrorC,ErrorC_rm,"convergence_rm_vs_normal.eps","Normal learning","Learning with negative weights")
    if(have_normal and have_ub and have_ub_nn):
        compare_covergence(MeanPrate,MeanPrate_ub,Error,Error_ub,MembraneVar,MembraneVar_ub,ErrorC,ErrorC_ub,"convergence_normal_vs_ub_vs_ub_nn.eps", "Normal learning","Learning using batched updates",include_bu_nn=True)