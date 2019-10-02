import numpy as np
from Learning import Learning
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
    # Initial FF matrix is sampled from a standard normal distribution
    F_initial = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    # FF matrix is normalized
    F_initial = utils.gamma*np.divide(F_initial, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F_initial**2, axis=0).reshape((1,utils.Nneuron)))))
    # Initial recurrent weights have small scales, except for the resets
    C_initial = -0.2*np.random.rand(utils.Nneuron, utils.Nneuron)-0.5*np.eye(utils.Nneuron)

    results = Learning(utils, F_initial, C_initial)

    ########## Dump the important files to Resources folder ##########
    for key in results:
        results[key].dump(os.path.join(resources, ("%s.dat" % key)))

    ########## Plotting ##########
    plot(results, utils, direc)

else:
    plot_from_resources(resources, utils, direc)
