"""
Created on Thu Sep 26 22:11:04 2019

@author: dzenn, julianb
"""
from helper import *
import numpy as np
from random import sample
from time import sleep, time
import operator
import rpyc
import itertools
import json
import os

import matplotlib.pyplot as plt

class SBSController():
    """
        Spike By Spike project Dynapse chip controller class
        
    """
    
    def __init__(self, start_neuron, chip_id, c, core_id, debug = False, base_isi=90, num_signals=2, clear_cam = True):
        
        self.start_neuron = start_neuron
        self.num_neurons = 20
        self.chip_id = chip_id
        self.core_id = core_id
        self.debug = debug
        self.c = c
        self.base_isi = base_isi
        self.clear_cam = clear_cam
        with open(os.path.join(os.getcwd(), "../parameters.param"), 'r') as f:
            self.parameters = json.load(f)
        
        
        try:
            print("rpyc ok")
            self.model = c.modules.CtxDynapse.model; print("model ok")
            self.vModel = self.c.modules.CtxDynapse.VirtualModel(); print("vModel ok")
            self.SynTypes = c.modules.CtxDynapse.DynapseCamType; print("SynTypes ok")
            self.dynapse = c.modules.CtxDynapse.dynapse; print("dynapse ok")
            self.connector = c.modules.NeuronNeuronConnector.DynapseConnector(); print("connector ok")
            self.fpga_spike_event = c.modules.CtxDynapse.FpgaSpikeEvent; print("Spike event ok")
        except AttributeError:
            print("Init failed: RPyC connection not active!")
            return
        
        self.num_signals = num_signals        
        self.spikegen = self.model.get_fpga_modules()[1]
        
        self.neurons = self.model.get_shadow_state_neurons()
        self.v_neurons = self.vModel.get_neurons()
        self.bias_group = self.model.get_bias_groups()[chip_id*4 + core_id]
        
        ADDR_NUM_BITS = 15
        ISI_NUM_BITS = 16
        self.max_fpga_len = 2**ADDR_NUM_BITS-1
        self.max_isi = 2**ISI_NUM_BITS-1 
        
        self.input_population = [self.v_neurons[idx] for idx in range(1, num_signals*2+1)]
        
        self.population = [n for n in self.neurons if n.get_chip_id()==self.chip_id
             and n.get_core_id()==self.core_id
             and n.get_neuron_id() >= self.start_neuron
             and n.get_neuron_id() < self.start_neuron + self.num_neurons]
        self.population_ids = [n.get_chip_id()*1024 + n.get_core_id()*256 + n.get_neuron_id() for n in self.population]

        if(self.debug and self.clear_cam):
            print("Clearing CAMs...")  
        if(self.clear_cam):          
            self.c.modules.CtxDynapse.dynapse.clear_cam(self.chip_id)
        if (self.debug):
            print("Finished clearing CAMs")
        
        
    @classmethod
    def from_default(self, clear_cam = True):
        c = rpyc.classic.connect("localhost", 1300)
        RPYC_TIMEOUT = 300 #defines a higher timeout
        c._config["sync_request_timeout"] = RPYC_TIMEOUT  # Set timeout to higher level

        return SBSController(start_neuron=1, chip_id=1, c=c, core_id=0, debug=True, clear_cam = clear_cam)
        
    def run_single_trial(self, plot_raster=False):
        """
        Runs a single input iteration and records the activity in the population.
        Optionally plots the recorded raster.
        
        """
        self.evt_filter = self.c.modules.CtxDynapse.BufferedEventFilter(self.model, self.population_ids)
        self.c.modules.CtxDynapse.dynapse.reset_timestamp()
        evts = self.evt_filter.get_events()
        self.spikegen.start()
        print("Running the trial...")
        sleep_time = self.spike_times[-1,1] / (1e+3*self.signal_length)
        sleep(sleep_time)
        evts = self.evt_filter.get_events()
        self.spikegen.stop()
        print("Trial finished")
        print("Binning the spikes")
        
        recorded_events = []
        
        if len(evts) != 0:
            for evt in evts:
                recorded_events.append([evt.neuron.get_id(), evt.timestamp])
                
        self.recorded_events = np.array(recorded_events)
        
        if plot_raster == True:
            self.plot_raster()
    
    
    def get_fpga_events(self, fpga_isi, fpga_nrn_ids):
        """ This function takes a list of events and neuron ids and returns an 
        object of FpgaSpikeEvent class.
        Args:
            fpga_isi     (list): list of isi 
            fpga_nrn_ids (list): list of neuron ids
        Returns:
            fpga_event   (FpgaSpikeEvent): ctxctl object
        """
        fpga_events = []
        for idx_isi, isi in enumerate(fpga_isi):
            fpga_event = self.fpga_spike_event()
            fpga_event.core_mask = 15
            fpga_event.target_chip = self.chip_id
            fpga_event.neuron_id = fpga_nrn_ids[idx_isi]
            fpga_event.isi = isi
            fpga_events.append(fpga_event)
        
        return fpga_events
    
    
    def spikes_to_isi(self, spike_times, neurons_id, use_microseconds=False):
        """ Function for coverting an array of spike times and array of corresponding
        neurons to inter spike intervals.
        Args:
            spike_times      (list): list of times. Either in milliseconds or microseconds
            neurons_id       (list): list of neuron ids. Should have same length as spike_times.
            use_microseconds (Bool): If set to True, will assume that spike_times are in millis.
                                     If set to False, will assume that spike_times are in micro sec.
        Returns:
            (signal_isi, neurons_id) (Tuple of lists): The signal ISI's and the corresponding neuron ids.
        """
        signal_isi = []
        for i in range(len(spike_times)):
            if i == 0 :
                signal_isi.append(spike_times[0])
            else:
                signal_isi.append(spike_times[i] - spike_times[i-1])
        signal_isi = np.asarray(signal_isi)
        if(use_microseconds):
            signal_isi = signal_isi * 1e3
        else: # Already in microseconds
            signal_isi = signal_isi

        # Avoid using neuron zero (because all neurons are connected to it)
        if(0 in neurons_id):
            neurons_id = neurons_id + 1 
        return (signal_isi, neurons_id)
    
    def load_spike_gen(self, fpga_events, isi_base, repeat_mode=False):
        """ This loads an FpgaSpikeEvent in the Spike Generator.
        Args:
            fpga_events (FpgaSpikeEvent):
            isibase     (isibase):
        """ 
        self.spikegen.set_variable_isi(True)
        self.spikegen.preload_stimulus(fpga_events)
        self.spikegen.set_isi_multiplier(isi_base)
        self.spikegen.set_repeat_mode(repeat_mode)

    def load_signal(self, x):
        self.signal_length = x.shape[1]
        print("Signal length is %d" % x.shape[1])
        ups = []; downs = []
        for i in range(x.shape[0]):
                tmp = signal_to_spike_refractory(1, np.linspace(0,len(x[i,:])-1,len(x[i,:])), x[i,:], 0.01*self.parameters["delta_modulator_threshold"], 0.01*self.parameters["delta_modulator_threshold"], 0.0001)
                ups.append(np.asarray(tmp[0]))
                downs.append(np.asarray(tmp[1]))

        self.spikes_up = np.asarray(ups)
        self.spikes_down = np.asarray(downs)

        self.spike_times = self.compile_preloaded_stimulus(dummy_neuron_id = 255)
        
        # Convert to ISI
        (signal_isi, neuron_ids) = self.spikes_to_isi(spike_times=self.spike_times[:,1], neurons_id=self.spike_times[:,0], use_microseconds=False)

        # Get the FPGA events
        fpga_events = self.get_fpga_events(signal_isi, neuron_ids)
        
        # Load spike gen
        self.load_spike_gen(fpga_events=fpga_events, isi_base=self.base_isi, repeat_mode=False)


    def set_feedforward_connection(self, F):
        self.F = F
        DWN1_weights = np.copy(self.F[:,1])
        DWN2_weights = np.copy(self.F[:,3])
        UP1_weights = np.copy(self.F[:,0])
        UP2_weights = np.copy(self.F[:,2])

        # F takes UP1, DWN1, UP2, DWN2
        # input comes as UP1, UP2, DWN1, DWN2
        self.F[:,0] = UP1_weights
        self.F[:,1] = UP2_weights
        self.F[:,2] = DWN1_weights
        self.F[:,3] = DWN2_weights 

        vn_list = []; pop_neur_list = []; syn_list = []

        for i in range(2*self.num_signals):
            for j in range(self.num_neurons):
                for k in range(abs(self.F[j,i])): # weight from in-i to pop-j
                    vn_list.append(self.input_population[i])
                    pop_neur_list.append(self.population[j])
                    if(self.F[j,i] > 0):
                        syn_list.append(self.SynTypes.FAST_EXC)
                    else:
                        syn_list.append(self.SynTypes.FAST_INH)

        self.connector.add_connection_from_list(vn_list, pop_neur_list, syn_list)
        self.model.apply_diff_state()
    
    def compile_preloaded_stimulus(self, dummy_neuron_id = 255):
        
        output_events = []
        for i in range(self.num_signals):
            for timestamp in self.spikes_up[i]:
                output_events.append([i+1, int(timestamp*1000)])
            for timestamp in self.spikes_down[i]:
                output_events.append([i+2, int(timestamp*1000)])
          
        output_events.sort(key=operator.itemgetter(1))
        output_events = np.insert(output_events, 0, [dummy_neuron_id,0], axis = 0)
        
        tmp_id = 1
        while tmp_id < len(output_events):
            if output_events[tmp_id,1] - output_events[tmp_id-1,1] > self.max_isi:
                output_events = np.insert(output_events, tmp_id, [dummy_neuron_id,output_events[tmp_id-1,1]+self.max_isi-1], axis = 0)
            tmp_id += 1
            
        return output_events
    
        
    def plot_raster(self):

        times = np.asarray(self.recorded_events[:,1] / 1000, dtype=int)
        plt.figure(figsize=(18, 6))
        plt.plot(times, self.recorded_events[:,0]-min(self.population_ids)+1, 'o', c='k', markersize=0.5)
        plt.ylim((0,self.parameters["Nneuron"]+1))
        plt.yticks(ticks=np.linspace(0,self.parameters["Nneuron"],int(self.parameters["Nneuron"]/2)+1))
        plt.title("DYNAPS spike train with initial discretized weights")
        plt.savefig("Resources/DYNAPS_initial_spikes.png")
        plt.show()
        
