import numpy as np
from Learning import Learning
from Utils import Utils
import sys
import os
import json
from plotting import *
import argparse
from AudioHelper import AudioHelper


TRAINING = False

parser = argparse.ArgumentParser(description='Training of optimal spike based signal representation on a neuromorphic chip.')
parser.add_argument('-t', help='test on new input', action='store_true')
parser.add_argument('-ua', help='compute rates and do updates using all neurons that spiked', action='store_true')
parser.add_argument('-d', help='discretize the weights', action='store_true')
parser.add_argument('-rm', help='remove positive recurrent connections', action='store_true')
parser.add_argument('-spiking', help='use spiking input', action='store_true')
parser.add_argument('-bnn', help='Use batched updates without normalizing', action='store_true')
parser.add_argument('-b', help='Use batched updates', action='store_true')
parser.add_argument('-audio', help='Use spoken MNIST as input', action='store_true')
parser.add_argument('-reinforce', help='Use Reinforcement learning based step size adaptation', action='store_true')

args = vars(parser.parse_args())
testing = args['t']
update_all = args['ua']
discretize = args['d']
remove_positive = args['rm']
use_spiking = args['spiking']
use_batched = args['b']
use_batched_nn = args['bnn']
use_audio = args['audio']
use_reinforcement = args['reinforce']

# Audio helper object
audio_helper = AudioHelper()

if(use_batched_nn):
    use_batched = True

########## Read parameter file #########

with open(os.path.join(os.getcwd(), "parameters.param"), 'r') as f:
    parameters = json.load(f)

with open(os.path.join(os.getcwd(), "audio_parameters.param"), 'r') as f:
    audio_parameters = json.load(f)


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

if(not use_audio):
    utils = Utils.from_json(parameters)
else:
    utils = Utils.from_json(audio_parameters)

if(not testing):
    # Initial FF matrix is sampled from a standard normal distribution
    F_initial = 0.5*np.random.randn(utils.Nx, utils.Nneuron)
    # FF matrix is normalized
    F_initial = utils.gamma*np.divide(F_initial, np.sqrt(np.matmul(np.ones((utils.Nx,1)), np.sum(F_initial**2, axis=0).reshape((1,utils.Nneuron)))))
    # Initial recurrent weights have small scales, except for the resets
    C_initial = -0.2*np.random.rand(utils.Nneuron, utils.Nneuron)-0.5*np.eye(utils.Nneuron)
    C_initial = -np.eye(utils.Nneuron)*0.5

    ########## Prepare weight matrix for the DYNAPS ##########
    if(not use_audio):
        M = np.asarray([[1, -1, 0, 0], [0, 0, 1, -1]])
    else:
        M = np.asarray([[1,-1]])
    FtM = np.matmul(F_initial.T, M)
    for i in range(FtM.shape[1]): # for all columns
        FtM[:,i] = FtM[:,i] / (max(FtM[:,i]) - min(FtM[:,i])) * 2*utils.dynapse_maximal_synapse_ff
    FtM = np.asarray(FtM, dtype=int)
    FtM.dump(os.path.join(os.getcwd(), "DYNAPS/Resources/DYNAPS/DYNAPS_F_disc.dat"))

    results = Learning(utils, F_initial, C_initial, update_all=update_all,
                    discretize_weights=discretize, remove_positive=remove_positive,
                    use_spiking=use_spiking, use_batched=use_batched,
                    use_batched_nn=use_batched_nn, use_audio=use_audio, audio_helper=audio_helper,
                    use_reinforcement=use_reinforcement)

    results["C_after"].dump(os.path.join(os.getcwd(), "DYNAPS/Resources/DYNAPS/DYNAPS_C_cont.dat"))
    results["xT"].dump(os.path.join(os.getcwd(), "DYNAPS/Resources/DYNAPS/bias_xT.dat"))
    results["OT_sim"].dump(os.path.join(os.getcwd(), "DYNAPS/Resources/DYNAPS/OT_sim.dat"))
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
        elif(use_audio):
            name = key + "_audio.dat"
        elif(use_reinforcement):
            name = key + "_reinforce.dat"
        else:
            name = key + ".dat"
        results[key].dump(os.path.join(resources, name))

    ########## Plotting ##########


plot_from_resources(resources, utils, direc, update_all=update_all,
                    discretize=discretize, remove_positive=remove_positive,
                    use_spiking=use_spiking,use_batched=use_batched,use_batched_nn=use_batched_nn,
                    use_audio=use_audio, audio_helper=audio_helper, use_reinforcement=use_reinforcement)
