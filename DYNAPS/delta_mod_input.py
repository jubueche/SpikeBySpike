from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import os

seed(43)
penable = False
utils = Utils.from_default()

if(not os.path.isdir(os.path.join(os.getcwd(), "Resources"))):
        os.mkdir(os.path.join(os.getcwd(), "Resources"))

F = np.load("Resources/F.dat", allow_pickle=True)

print(F.T)
print("")

conn_x_high = []
conn_x_down = []

for i in range(0,F.shape[0]):
        tmp_up = np.copy(F.T[:,i]); tmp_down = np.copy(F.T[:,i])
        tmp_up[tmp_up < 0] = 0 ; tmp_down[tmp_down >= 0] = 0; tmp_down *= -1
        tmp_up = tmp_up / (max(tmp_up)-min(tmp_up))*10 # Scale from 0 to 10
        tmp_down = tmp_down / (max(tmp_down)-min(tmp_down))*10
        conn_x_high.append(tmp_up)
        conn_x_down.append(tmp_down)  
        
print(np.asarray(conn_x_high).T)
print(np.asarray(conn_x_down).T)

for i in range(F.shape[0]):
        conn_x_high[i].dump(os.path.join("Resources", ("x%d_up.dat" % i)))
        conn_x_down[i].dump(os.path.join("Resources", ("x%d_down.dat" % i)))

# Show a heat map of F
utils.save_F("Resources/F.png", F)

# Get the signal
x = utils.get_matlab_like_input()

x1 = x[0,:]
x2 = x[1,:]

thresh = 0.05 #! Move to utils
ups = []; downs = []; ups_isi = []; downs_isi = []
for i in range(x.shape[0]):
        tmp = utils.signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], thresh, thresh, 0.0001)
        ups.append(tmp[0])
        downs.append(tmp[1])
        tmp_up_isi = utils.spikes_to_isi(tmp[0], 1*np.ones(utils.duration), use_microseconds=True)[0]
        ups_isi.append(tmp_up_isi)
        tmp_down_isi = utils.spikes_to_isi(tmp[1], 1*np.ones(utils.duration), use_microseconds=True)[0]
        downs_isi.append(tmp_down_isi)
        tmp_up_isi.dump(os.path.join("Resources", ("up_%d_isi.dat" % i)))
        tmp_down_isi.dump(os.path.join("Resources", ("down_%d_isi.dat" % i)))

###### Plotting spike trains ###### 
if(penable):
        utils.plot_delta_spike_trains(x, ups, downs)