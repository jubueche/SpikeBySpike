import numpy as np
from Learning import Learning, spiking_to_continous
from Utils import Utils
import sys
import os
import json
from plotting import *

TRAINING = True

########## Read parameter file #########

if(sys.argv[1] != None):
    saveFolder = sys.argv[1]
else:
    raise SystemExit('Error: Please specify folder to save data in.')
if(sys.argv[2] != None):
    with open(os.path.join(os.getcwd(), sys.argv[2]), 'r') as f:
        parameters = json.load(f)
else:
    raise SystemExit('Error: Please add parameters file.')

######### Create save folder and resources folder ##########

resources = os.path.join(os.getcwd(), "DYNAPS/Resources")
if saveFolder:
    direc = os.path.join(os.getcwd(), saveFolder)
else:
    direc = os.path.join(os.getcwd(), "log")
if(not os.path.exists(direc)):
    os.mkdir(direc)
if(not os.path.exists(resources)):
    os.mkdir(resources)

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

utils = Utils.from_json(parameters)

if(TRAINING):
    M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    # Initial FF matrix is sampled from a standard normal distribution
    F_initial = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    # FF matrix is normalized
    F_initial = utils.gamma*np.divide(F_initial, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F_initial**2, axis=0).reshape((1,utils.Nneuron)))))

    # Initial recurrent weights have small scales, except for the resets
    #! Uncomment for true initialization
    #C_initial = -0.2*np.random.rand(utils.Nneuron, utils.Nneuron)-0.5*np.eye(utils.Nneuron)

    # Changed to diagonal negative thresholds
    C_initial = -utils.Thresh*np.eye(utils.Nneuron)

    ########## Prepare weight matrix for the DYNAPS ##########
    FtM = np.matmul(F_initial.T, M)
    for i in range(FtM.shape[1]): # for all columns
        FtM[:,i] = FtM[:,i] / (max(FtM[:,i]) - min(FtM[:,i])) * 2*utils.dynapse_maximal_synapse
    FtM = np.asarray(FtM, dtype=int)
    FtM.dump(os.path.join(os.getcwd(), "DYNAPS/Resources/DYNAPS_F.dat"))

    results = Learning(utils, F_initial, C_initial)

    ########## Dump the important files to Resources folder ##########
    for key in results:
        results[key].dump(os.path.join(resources, ("%s.dat" % key)))

    ########## Plotting ##########
    plot(results, utils, direc)

else:

   # spiking_to_continous(utils)

    plot_from_resources(resources, utils, direc)
