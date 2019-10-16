"""
Before running this cell:
1) $ cd ~/Documents/NCS/cortexcontrol
2) $ sudo ./cortexcontrol
3) $ exec(open("./start_rpyc.py").read())
"""
import sbs_dynapse_controller
import sys
import os
import numpy as np
from DYNAPS_Learning import Learning
sys.path.append(os.path.join(os.getcwd(),"../"))
from Utils import Utils
from plotting import plot
import traceback

TRAINING = True

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
print("Setting DYNAPS parameters...")
sbs.c.execute("exec(open('" + self_path + "/Resources/balanced.py').read())")
sbs.model.apply_diff_state()

utils = Utils.from_json(sbs.parameters)

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

resources = os.path.join(os.getcwd(), "Resources/DYNAPS")
if(not os.path.exists(resources)):
    os. mkdir(resources)

if(TRAINING):
    M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    # Initial F matrix is sampled from a standard normal distribution
    F = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    # F matrix is normalized
    F = utils.gamma*np.divide(F, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F**2, axis=0).reshape((1,utils.Nneuron)))))

    # Changed to diagonal negative thresholds
    C_initial = -utils.Thresh*np.eye(utils.Nneuron)

    ########## Prepare weight matrix for the DYNAPS ##########
    FtM = np.matmul(F.T, M)
    for i in range(FtM.shape[1]): # for all columns
        FtM[:,i] = FtM[:,i] / (max(FtM[:,i]) - min(FtM[:,i])) * 2*utils.dynapse_maximal_synapse_ff
    FtM = np.asarray(FtM, dtype=int)

    try:
        results = Learning(sbs, utils, F, FtM, C_initial)
        for key in results:
            results[key].dump(os.path.join(resources, ("%s.dat" % key)))

        ########## Plotting ##########
        plot(results, utils, resources)

    except Exception:
        traceback.print_exc()
        clean_up_conn(sbs)
    except KeyboardInterrupt:
        clean_up_conn(sbs)
    except:
        clean_up_conn(sbs)
        print("Cleaned up connections.")
