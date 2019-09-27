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
        self.spikes_0_up = np.load("Resources/x0_up.dat")
        self.spikes_0_down = np.load("Resources/x0_down.dat")
        self.spikes_1_up = np.load("Resources/x1_up.dat")
        self.spikes_1_down = np.load("Resources/x1_down.dat")
        
        fpga_evts = self.compile_preloaded_stimulus(dummy_neuron_id = 255)
        
        ##TODO: Convert timestamps to ISI
        
        
#        for isi in self.isi_1_up:
#            fpga_event = self.c.modules.CtxDynapse.FpgaSpikeEvent()
#            fpga_event.core_mask = 15
#            fpga_event.target_chip = self.chip_id
#            fpga_event.neuron_id = 1
#            fpga_event.isi = int(isi)
#            fpga_evts.append(fpga_event)
#        
#        
#        self.spikegen.set_variable_isi(True)
#        self.spikegen.preload_stimulus([fpga_event])
#        self.spikegen.set_repeat_mode(False)
    
    def compile_preloaded_stimulus(self, dummy_neuron_id = 255):
        
        output_events = []
        for timestamp in self.spikes_0_up:
            output_events.append([1, int(timestamp*1000)])
        for timestamp in self.spikes_0_down:
            output_events.append([2, int(timestamp*1000)])
        for timestamp in self.spikes_1_up:
            output_events.append([3, int(timestamp*1000)])
        for timestamp in self.spikes_1_down:
            output_events.append([4, int(timestamp*1000)])
            
        output_events = output_events[output_events[:,1].argsort()]
        output_events = np.insert(output_events, 0, [dummy_neuron_id,0], axis = 0)
        
        tmp_id = 1
        while tmp_id < len(output_events):
            if output_events[tmp_id,1] - output_events[tmp_id-1,1] > self.max_isi:
                output_events = np.insert(output_events, tmp_id, [dummy_neuron_id,output_events[tmp_id-1,1]+self.max_isi-1], axis = 0)
            tmp_id += 1
            
        return output_events
        
    def plot_raster(self):
        pass