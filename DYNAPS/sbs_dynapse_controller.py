#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:11:04 2019

@author: dzenn
"""

import numpy as np
from random import sample

from time import sleep, time



class SBSController():
    """
        Spike By Spike project Dynapse chip controller class
        
    """
    
    def __init__(self, start_neuron, chip_id, c, core_id, debug = False):
        
        self.start_neuron = start_neuron
        self.num_neurons = 20
        self.chip_id = chip_id
        self.core_id = core_id
        self.debug = debug
        self.c = c
        
        try:
            print("rpyc ok")
            self.model = c.modules.CtxDynapse.model
            print("model ok")
            self.vModel = self.c.modules.CtxDynapse.VirtualModel()
            print("vModel ok")
            self.connector = c.modules.NeuronNeuronConnector.DynapseConnector()

            print("connector ok")
            self.SynTypes = c.modules.CtxDynapse.DynapseCamType
            print("SynTypes ok")
            self.dynapse = c.modules.CtxDynapse.dynapse
            print("dynapse ok")
        except AttributeError:
            print("Init failed: RPyC connection not active!")
            return
        
                
        self.spikegen = self.model.get_fpga_modules()[1]
        
        self.neurons = self.model.get_shadow_state_neurons()
        self.v_neurons = self.vModel.get_neurons()
        self.bias_group = self.model.get_bias_groups()[chip_id*4 + core_id]
        
        ADDR_NUM_BITS = 15
        ISI_NUM_BITS = 16
        self.max_fpga_len = 2**ADDR_NUM_BITS-1
        self.max_isi = 2**ISI_NUM_BITS-1 
        
        self.population = [n for n in self.neurons if n.get_chip_id()==self.chip_id
             and n.get_core_id()==self.core_id
             and n.get_neuron_id() >= self.start_neuron
             and n.get_neuron_id() < self.start_neuron + self.num_neurons]
        self.population_ids = [n.get_chip_id()*1024 + n.get_core_id()*256 + n.get_neuron_id() for n in self.population]
        
        
    def run_single_trial(self):
        
        pass
        
    def load_resources(self):
        """
            Loading all resource files: input weights and preloaded spiketrains
        """
        
        self.input_weights = np.load("Resources/F.dat")
        self.isi_1_up = np.load("Resources/up_1_isi.dat")
        self.isi_1_down = np.load("Resources/down_1_isi.dat")
        self.isi_2_up = np.load("Resources/up_2_isi.dat")
        self.isi_2_down = np.load("Resources/down_2_isi.dat")
        
        fpga_evts = []
        
        for isi in self.isi_1_up:
            fpga_event = self.c.modules.CtxDynapse.FpgaSpikeEvent()
            fpga_event.core_mask = 15
            fpga_event.target_chip = self.chip_id
            fpga_event.neuron_id = 1
            fpga_event.isi = int(isi)
            fpga_evts.append(fpga_event)
            
        for isi in self.isi_1_down:
            fpga_event = self.c.modules.CtxDynapse.FpgaSpikeEvent()
            fpga_event.core_mask = 15
            fpga_event.target_chip = self.chip_id
            fpga_event.neuron_id = 2
            fpga_event.isi = int(isi)
            fpga_evts.append(fpga_event)
            
        for isi in self.isi_2_up:
            fpga_event = self.c.modules.CtxDynapse.FpgaSpikeEvent()
            fpga_event.core_mask = 15
            fpga_event.target_chip = self.chip_id
            fpga_event.neuron_id = 3
            fpga_event.isi = int(isi)
            fpga_evts.append(fpga_event)
            
        for isi in self.isi_2_down:
            fpga_event = self.c.modules.CtxDynapse.FpgaSpikeEvent()
            fpga_event.core_mask = 15
            fpga_event.target_chip = self.chip_id
            fpga_event.neuron_id = 4
            fpga_event.isi = int(isi)
            fpga_evts.append(fpga_event)
        
        
        self.spikegen.set_variable_isi(True)
        self.spikegen.preload_stimulus([fpga_event])
        self.spikegen.set_repeat_mode(False)

        
    def plot_raster(self):
        pass