import numpy as np
from Learning import Learning
from Utils import Utils
import sys
import os
import json
from plotting import *
import argparse

TRAINING = False

parser = argparse.ArgumentParser(description='Training of optimal spike based signal representation on a neuromorphic chip.')
parser.add_argument('-t', help='test on new input', action='store_true')
parser.add_argument('-ua', help='compute rates and do updates using all neurons that spiked', action='store_true')
parser.add_argument('-d', help='discretize the weights', action='store_true')
parser.add_argument('-rm', help='remove positive recurrent connections', action='store_true')
parser.add_argument('-spiking', help='use spiking input', action='store_true')
parser.add_argument('-bnn', help='Use batched updates without normalizing', action='store_true')
parser.add_argument('-b', help='Use batched updates', action='store_true')
args = vars(parser.parse_args())
testing = args['t']
update_all = args['ua']
discretize = args['d']
remove_positive = args['rm']
use_spiking = args['spiking']
use_batched = args['b']
use_batched_nn = args['bnn']
if(use_batched_nn):
    use_batched = True

########## Read parameter file #########

with open(os.path.join(os.getcwd(), "parameters.param"), 'r') as f:
    parameters = json.load(f)
saveFolder = "Data"

######### Create save folder and resources folder ##########

resources = os.path.join(os.getcwd(), "DYNAPS/Resources/Simulation")
direc = os.path.join(os.getcwd(), saveFolder)
if(not os.path.exists(direc)):
    os.mkdir(direc)
if(not os.path.exists(resources)):
    os.mkdir(resources)

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

utils = Utils.from_json(parameters)

if(not testing):
    # Initial FF matrix is sampled from a standard normal distribution
    F_initial = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    # FF matrix is normalized
    F_initial = utils.gamma*np.divide(F_initial, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F_initial**2, axis=0).reshape((1,utils.Nneuron)))))
    # Initial recurrent weights have small scales, except for the resets
    C_initial = -0.2*np.random.rand(utils.Nneuron, utils.Nneuron)-0.5*np.eye(utils.Nneuron)

    results = Learning(utils, F_initial, C_initial, update_all=update_all,
                    discretize_weights=discretize, remove_positive=remove_positive,
                    use_spiking=use_spiking, use_batched=use_batched, use_batched_nn=use_batched_nn)

    ########## Dump the important files to Resources folder ##########
    for key in results:
        if(update_all):
            name = key + "_ua.dat"
        elif(discretize):
            name = key + "_d.dat"
        elif(remove_positive):
            name = key + "_rm.dat"
        elif(use_spiking):
            name = key + "_us.dat"
        elif(use_batched_nn):
            name = key + "_ub_nn.dat"
        elif(use_batched):
            name = key + "_ub.dat"
        else:
            name = key + ".dat"
        results[key].dump(os.path.join(resources, name))

    ########## Plotting ##########


plot_from_resources(resources, utils, direc, update_all=update_all,
                    discretize=discretize, remove_positive=remove_positive,
                    use_spiking=use_spiking,use_batched=use_batched,use_batched_nn=use_batched_nn)
