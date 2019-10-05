from helper import *
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
from runnet import runnet, runnet_spike_input

np.random.seed(42)

duration = 1000
maximal_input_syn = 10

with open(os.path.join(os.getcwd(), "../parameters.param"), 'r') as f:
        parameters = json.load(f)

if(not os.path.isdir(os.path.join(os.getcwd(), "Resources"))):
        SystemError("Error: No Resource folder found.")


# Get the signal
x = get_input(A=parameters["A"], Nx=parameters["Nx"], dt=parameters["dt"], lam=parameters["lam"], duration=duration)
x.dump("Resources/x_in.dat")

ups = []; downs = []
for i in range(x.shape[0]):
        tmp = signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], parameters["delta_modulator_threshold"], parameters["delta_modulator_threshold"], 0.0001)
        ups.append(np.asarray(tmp[0]))
        downs.append(np.asarray(tmp[1]))

ups = np.asarray(ups)
downs = np.asarray(downs)

for idx, (up, down) in enumerate((ups, downs)):
        up.dump(os.path.join("Resources",("x%d_up.dat" % idx)))
        down.dump(os.path.join("Resources",("x%d_down.dat" % idx)))

OT_up = np.zeros((parameters["Nx"], parameters["Ntime"]))
OT_down = np.zeros((parameters["Nx"], parameters["Ntime"]))
for i in range(parameters["Nx"]):
        OT_up[i,np.asarray(ups[i], dtype=int)] = 1
        OT_down[i,np.asarray(downs[i], dtype=int)] = 1

###### Plotting spike trains ###### 
plt.figure(figsize=(12, 12))

plt.subplot(611)
plt.title("Signal 1")
plt.plot(x[0,:], c='r')
plt.xlim((0,duration))
plt.subplot(612)
plt.plot(ups[0], np.zeros(len(ups[0])), 'o', c='k', markersize=1)
plt.xlim((0,duration))
plt.subplot(613)
plt.plot(downs[0], np.zeros(len(downs[0])), 'o', c='k', markersize=1)
plt.xlim((0,duration))

plt.subplot(614)
plt.title("Signal 2")
plt.plot(x[1,:], c='r')
plt.xlim((0,duration))
plt.subplot(615)
plt.plot(ups[1], np.zeros(len(ups[1])), 'o', c='k', markersize=1)
plt.xlim((0,duration))
plt.subplot(616)
plt.plot(downs[1], np.zeros(len(downs[1])), 'o', c='k', markersize=1)
plt.xlim((0,duration))

plt.tight_layout()
plt.savefig("Resources/signal_transformation.png")
plt.show()


########## How does the spike train look when we run it on the initial matrix? ##########
Fi = np.load("Resources/Fi.dat", allow_pickle=True)
Ci = np.load("Resources/Ci.dat", allow_pickle=True)

DYNAPSconn_x_up_spikes = np.load("Resources/DYNAPSconn_x_up_spikes.dat", allow_pickle=True)
DYNAPSconn_x_down_spikes = np.load("Resources/DYNAPSconn_x_down_spikes.dat", allow_pickle=True)
conn_x_up_spikes = np.load("Resources/conn_x_up_spikes.dat", allow_pickle=True)
conn_x_down_spikes = np.load("Resources/conn_x_down_spikes.dat", allow_pickle=True)

for i in range(parameters["Nx"]):
        DYNAPSconn_x_up_spikes[i].dump(("Resources/DYNAPSconn_x%d_up_spikes.dat" % i))
        DYNAPSconn_x_down_spikes[i].dump(("Resources/DYNAPSconn_x%d_down_spikes.dat" % i))        

(_,OT,_) = runnet_spike_input(dt=parameters["dt"], lam=parameters["lam"], conn_x_high=conn_x_up_spikes, conn_x_down=conn_x_down_spikes,
                         OT_up=OT_up, OT_down=OT_down, C=Ci, Nneuron=parameters["Nneuron"],Ntime=duration, Thresh=parameters["Thresh"])

coordinates = np.nonzero(OT)

plt.figure(figsize=(18, 6))
plt.scatter(coordinates[1], coordinates[0]+1, marker='o', s=0.5, c='k')
plt.ylim((0,parameters["Nneuron"]+1))
plt.yticks(ticks=np.linspace(0,parameters["Nneuron"],int(parameters["Nneuron"]/2)+1))
plt.title("Population spike train using initial weights")
plt.savefig("Resources/initial_pop_spikes.png")
plt.show()