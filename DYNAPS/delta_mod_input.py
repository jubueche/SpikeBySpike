from helper import *
import numpy as np
import os
import json
import matplotlib.pyplot as plt

duration = 1000

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
        tmp_up = tmp_up / (max(tmp_up)-min(tmp_up))*10 # Scale from 0 to 10
        tmp_down = tmp_down / (max(tmp_down)-min(tmp_down))*10
        conn_x_high.append(tmp_up)
        conn_x_down.append(tmp_down)  
        
for i in range(F.shape[0]):
        conn_x_high[i].dump(os.path.join("Resources", ("conn_x%d_up.dat" % i)))
        conn_x_down[i].dump(os.path.join("Resources", ("conn_x%d_down.dat" % i)))


# Get the signal
x = get_input(A=parameters["A"], Nx=parameters["Nx"], dt=parameters["dt"], lam=parameters["lam"], duration=duration)

threshold = 0.5
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