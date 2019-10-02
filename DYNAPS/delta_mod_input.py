from helper import *
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
from runnet import runnet

np.random.seed(42)

duration = 1000
threshold = 20
maximal_input_syn = 10

with open(os.path.join(os.getcwd(), "../parameters.param"), 'r') as f:
        parameters = json.load(f)


if(not os.path.isdir(os.path.join(os.getcwd(), "Resources"))):
        SystemError("Error: No Resource folder found.")

F = np.load("Resources/F_after.dat", allow_pickle=True)

conn_x_high = []
conn_x_down = []

for i in range(0,F.shape[0]):
        tmp_up = np.copy(F.T[:,i]); tmp_down = np.copy(F.T[:,i])
        tmp_up[tmp_up < 0] = 0 ; tmp_down[tmp_down >= 0] = 0; tmp_down *= -1
        tmp_up = tmp_up / (max(tmp_up)-min(tmp_up))*maximal_input_syn # Scale from 0 to 10
        tmp_down = tmp_down / (max(tmp_down)-min(tmp_down))*maximal_input_syn
        conn_x_high.append(tmp_up)
        conn_x_down.append(tmp_down)  
        
for i in range(F.shape[0]):
        conn_x_high[i].dump(os.path.join("Resources", ("conn_x%d_up.dat" % i)))
        conn_x_down[i].dump(os.path.join("Resources", ("conn_x%d_down.dat" % i)))


# Get the signal
x = get_input(A=parameters["A"], Nx=parameters["Nx"], dt=parameters["dt"], lam=parameters["lam"], duration=duration)
x.dump("Resources/x_in.dat")

ups = []; downs = []
for i in range(x.shape[0]):
        tmp = signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], threshold, threshold, 0.0001)
        ups.append(np.asarray(tmp[0]))
        downs.append(np.asarray(tmp[1]))

ups = np.asarray(ups)
downs = np.asarray(downs)

for idx, (up, down) in enumerate((ups, downs)):
        up.dump(os.path.join("Resources",("x%d_up.dat" % idx)))
        down.dump(os.path.join("Resources",("x%d_down.dat" % idx)))


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


(_,OT,_) = runnet(dt=parameters["dt"], lam=parameters["lam"], F=Fi, Input=x, C=Ci,
                        Nneuron=parameters["Nneuron"],Ntime=duration, Thresh=parameters["Thresh"])


coordinates = np.nonzero(OT)

plt.figure(figsize=(18, 6))
plt.scatter(coordinates[1], coordinates[0]+1, marker='o', s=0.5, c='k')
plt.ylim((0,parameters["Nneuron"]+1))
plt.yticks(ticks=np.linspace(0,parameters["Nneuron"],int(parameters["Nneuron"]/2)+1))
plt.title("Population spike train using initial weights")
plt.savefig("Resources/initial_pop_spikes.png")
plt.show()