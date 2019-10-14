"""
Before running this cell:
1) $ cd ~/Documents/NCS/cortexcontrol
2) $ sudo ./cortexcontrol
3) $ exec(open("./start_rpyc.py").read())
"""
import sbs_dynapse_controller
import sys
import numpy as np

clear_cam = False

if(len(sys.argv) > 1):
    cc = sys.argv[1]
    if(cc == "True"):
        clear_cam = True

def clean_up_conn(sbs):
    for n in sbs.population:
        sbs.connector.remove_receiving_connections(n)

    sbs.c.close()

self_path = sbs_dynapse_controller.__file__
self_path = self_path[0:self_path.rfind('/')]

sbs = sbs_dynapse_controller.SBSController.from_default(clear_cam = clear_cam)
x = np.load("Resources/x_in.dat", allow_pickle=True)
sbs.load_signal(x)
F = np.load("Resources/DYNAPS_F.dat", allow_pickle=True)
sbs.set_feedforward_connection(F)

sbs.c.execute("exec(open('" + self_path + "/Resources/balanced.py').read())")

sbs.model.apply_diff_state()

try:
    sbs.run_single_trial(plot_raster=True)
except:
    print("No spikes.")
    clean_up_conn(sbs)
else:
    clean_up_conn(sbs)
