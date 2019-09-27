from utils_input import UtilsInput
import numpy as np
from brian2 import *
import os

seed(43)
penable = False
utils = UtilsInput.from_default()

if(not os.path.isdir(os.path.join(os.getcwd(), "Resources"))):
        os.mkdir(os.path.join(os.getcwd(), "Resources"))

F = np.load("Resources/F.dat", allow_pickle=True)

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
        conn_x_high[i].dump(os.path.join("Resources", ("x%d_up.dat" % i)))
        conn_x_down[i].dump(os.path.join("Resources", ("x%d_down.dat" % i)))

# Show a heat map of F
utils.save_F("Resources/F.png", F)

# Get the signal
x = utils.get_matlab_like_input()


ups = []; downs = []
for i in range(x.shape[0]):
        tmp = utils.signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], utils.threshold, utils.threshold, 0.0001)
        ups.append(np.asarray(tmp[0]))
        downs.append(np.asarray(tmp[1]))

ups = np.asarray(ups)
downs = np.asarray(downs)

for idx, (up, down) in enumerate((ups, downs)):
        up.dump(os.path.join("Resources",("x%d_up.dat" % idx)))
        down.dump(os.path.join("Resources",("x%d_down.dat" % idx)))


###### Plotting spike trains ###### 
if(penable):
        utils.plot_delta_spike_trains(x, ups, downs)