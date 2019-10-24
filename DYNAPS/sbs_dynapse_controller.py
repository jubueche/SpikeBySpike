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
import math
import warnings

import matplotlib.pyplot as plt

class SBSController():
    """
        Spike By Spike project Dynapse chip controller class
        
    """
    
    def __init__(self, start_neuron, chip_id, c, core_id, debug = False, base_isi=90, num_signals=2, clear_cam = True):
        
        self.start_neuron = start_neuron
        self.chip_id = chip_id
        self.core_id = core_id
        self.debug = debug
        self.c = c
        self.base_isi = base_isi
        self.clear_cam = clear_cam
        with open(os.path.join(os.getcwd(), "../parameters.param"), 'r') as f:
            self.parameters = json.load(f)
        self.num_neurons = self.parameters["Nneuron"]
        self.C = np.zeros((self.num_neurons, self.num_neurons)).astype(np.int) # Initialized to 0
        
        try:
            print("RPYC: OK")
            self.model = c.modules.CtxDynapse.model; print("Model: OK")
            self.groups = self.model.get_bias_groups(); print("Bias Groups: OK")
            self.vModel = self.c.modules.CtxDynapse.VirtualModel(); print("vModel: OK")
            self.SynTypes = c.modules.CtxDynapse.DynapseCamType; print("SynTypes: OK")
            self.dynapse = c.modules.CtxDynapse.dynapse; print("DYNAPS: OK")
            self.connector = c.modules.NeuronNeuronConnector.DynapseConnector(); print("Connector: OK")
            self.fpga_spike_event = c.modules.CtxDynapse.FpgaSpikeEvent; print("Spike event: OK")
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
        if(self.debug and not self.clear_cam):
            print("Not clearing CAMs.")
        if(self.clear_cam):          
            self.c.modules.CtxDynapse.dynapse.clear_cam(self.chip_id)
        if (self.debug and self.clear_cam):
            print("Finished clearing CAMs.")
        
        
    @classmethod
    def from_default(self, clear_cam = True, debug = False):
        c = rpyc.classic.connect("localhost", 1300)
        RPYC_TIMEOUT = 300 #defines a higher timeout
        c._config["sync_request_timeout"] = RPYC_TIMEOUT  # Set timeout to higher level

        return SBSController(start_neuron=1, chip_id=1, c=c, core_id=0, debug=debug, clear_cam = clear_cam)


    def execute(self):

        self.evt_filter = self.c.modules.CtxDynapse.BufferedEventFilter(self.model, self.population_ids)
        self.c.modules.CtxDynapse.dynapse.reset_timestamp()
        evts = self.evt_filter.get_events()
        
        self.spikegen.start()
        sleep_time = self.spike_times[-1,1] / (1e+6)
        sleep(sleep_time)
        evts = self.evt_filter.get_events()
        self.spikegen.stop()

        self.evt_filter.clear()
        
        recorded_events = []
        
        if len(evts) != 0:
            for evt in evts:
                recorded_events.append([evt.neuron.get_id(), evt.timestamp])
                
        self.recorded_events = np.array(recorded_events)
        times = np.asarray(self.recorded_events[:,1] / 1000, dtype=int)
        neuron_ids = self.recorded_events[:,0]-min(self.population_ids) # Starting at 0
        O_DYNAPS_sbs = np.zeros((self.num_neurons, self.signal_length))

        if(max(times) >= self.signal_length):
            neuron_ids = [ni for (idx,ni) in enumerate(neuron_ids) if times[idx] < self.signal_length]
            times = [t for t in times if t < self.signal_length]

        O_DYNAPS_sbs[neuron_ids, times] = 1

        """if(self.debug):
            coordinates = np.nonzero(O_DYNAPS_sbs)
            plt.figure(figsize=(18, 6))
            plt.scatter(coordinates[1], coordinates[0]+1, marker='o', s=0.5, c='k')
            plt.ylim((0,self.num_neurons+1))
            plt.yticks(ticks=np.linspace(0,self.num_neurons,int(self.num_neurons/2)+1))
            plt.show()"""

        return O_DYNAPS_sbs

    
    
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
    
    
    def spikes_to_isi(self, spike_times, n_ids, use_microseconds=False):
        """ Function for coverting an array of spike times and array of corresponding
        neurons to inter spike intervals.
        Args:
            spike_times      (list): list of times. Either in milliseconds or microseconds
            n_ids       (list): list of neuron ids. Should have same length as spike_times.
            use_microseconds (Bool): If set to True, will assume that spike_times are in millis.
                                     If set to False, will assume that spike_times are in micro sec.
        Returns:
            (signal_isi, n_ids) (Tuple of lists): The signal ISI's and the corresponding neuron ids.
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
        if(0 in n_ids):
            n_ids = n_ids + 1 
        return (signal_isi, n_ids)
    
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

    def load_signal(self, X, delta_mod_thresh_up, delta_mod_thresh_dwn):
        self.signal_length = X.shape[1]
        ups = []; downs = []
        for i in range(X.shape[0]):
            tmp = signal_to_spike_refractory(1, np.linspace(0,len(X[i,:])-1,len(X[i,:])), np.copy(X[i,:]), delta_mod_thresh_up, delta_mod_thresh_dwn, 0.0001)
            ups.append(np.asarray(np.copy(tmp[0])))
            downs.append(np.asarray(np.copy(tmp[1])))

        self.spikes_up = np.copy(np.asarray(ups))
        self.spikes_down = np.copy(np.asarray(downs))

        """if(self.debug):
            duration = self.signal_length
            plt.figure(figsize=(12, 12))

            plt.subplot(611)
            plt.title("Signal 1")
            plt.plot(X[0,:], c='r')
            plt.xlim((0,duration))
            plt.subplot(612)
            plt.plot(ups[0], np.zeros(len(ups[0])), 'o', c='k', markersize=1)
            plt.xlim((0,duration))
            plt.subplot(613)
            plt.plot(downs[0], np.zeros(len(downs[0])), 'o', c='k', markersize=1)
            plt.xlim((0,duration))

            plt.subplot(614)
            plt.title("Signal 2")
            plt.plot(X[1,:], c='r')
            plt.xlim((0,duration))
            plt.subplot(615)
            plt.plot(ups[1], np.zeros(len(ups[1])), 'o', c='k', markersize=1)
            plt.xlim((0,duration))
            plt.subplot(616)
            plt.plot(downs[1], np.zeros(len(downs[1])), 'o', c='k', markersize=1)
            plt.xlim((0,duration))

            plt.tight_layout()
            plt.show()"""


        self.spike_times = self.compile_preloaded_stimulus(dummy_neuron_id = 255)
        
        # Convert to ISI
        (signal_isi, neuron_ids) = self.spikes_to_isi(spike_times=self.spike_times[:,1], n_ids=self.spike_times[:,0], use_microseconds=False)

        # Get the FPGA events
        fpga_events = self.get_fpga_events(signal_isi, neuron_ids)

        # Load spike gen
        self.load_spike_gen(fpga_events=fpga_events, isi_base=self.base_isi, repeat_mode=False)

    def compile_preloaded_stimulus(self, dummy_neuron_id = 255):
        
        output_events = []
        for i in range(self.num_signals):
            for timestamp in self.spikes_up[i]:
                output_events.append([self.num_signals*i+1, int(timestamp*1000)])
            for timestamp in self.spikes_down[i]:
                output_events.append([self.num_signals*i+2, int(timestamp*1000)])

        output_events.sort(key=operator.itemgetter(1))
        output_events = np.insert(output_events, 0, [dummy_neuron_id,0], axis = 0)

        tmp_id = 1
        while tmp_id < len(output_events):
            if output_events[tmp_id,1] - output_events[tmp_id-1,1] > self.max_isi:
                output_events = np.insert(output_events, tmp_id, [dummy_neuron_id,output_events[tmp_id-1,1]+self.max_isi-1], axis = 0)
            tmp_id += 1
        
        return output_events

    def set_feedforward_connection(self, F_disc):
        self.F = F_disc

        vn_list = []; pop_neur_list = []; syn_list = []

        for i in range(2*self.num_signals):
            for j in range(self.num_neurons):
                for k in range(abs(self.F[j,i])): # weight from in-i to pop-j
                    vn_list.append(self.input_population[i])
                    pop_neur_list.append(self.population[j])
                    if(self.F[j,i] > 0):
                        syn_list.append(self.SynTypes.FAST_EXC)
                    elif(self.F[j,i] < 0):
                        syn_list.append(self.SynTypes.FAST_INH)

        self.connector.add_connection_from_list(vn_list, pop_neur_list, syn_list)
        self.model.apply_diff_state()

    """
    Pre: self.C must be discretized and have data type integer. It shall hold new value of C_disc
    """
    def set_recurrent_connection(self):

        for n in self.population:
            self.connector.remove_receiving_connections(n)
        self.model.apply_diff_state()

        self.set_feedforward_connection(self.F)
        self.model.apply_diff_state()

        pop_neur_source_list = []; pop_neur_target_list = []; syn_list = []
        # Rows in Omega are the targets and columns are source. C(i,j) is the weight from j -> i
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if(not (i == j)): # No autapses
                    for k in range(abs(self.C[i,j])):
                        # Source: j, Target: i
                        pop_neur_source_list.append(self.population[j])
                        pop_neur_target_list.append(self.population[i])
                        if(self.C[i,j] > 0):
                            syn_list.append(self.SynTypes.FAST_EXC)
                        elif(self.C[i,j] < 0):
                            syn_list.append(self.SynTypes.FAST_INH)

        self.connector.add_connection_from_list(pop_neur_source_list, pop_neur_target_list, syn_list)
        self.model.apply_diff_state()


    """
    Pre: Weight matrix must be discretized and sum of each row must be below maximum number of neurons allowed.
    """
    def set_recurrent_weight_directly(self, C_discrete):
        number_available_per_neuron = 62 - np.sum(np.abs(self.F), axis=1)
        assert (number_available_per_neuron - np.sum(np.abs(C_discrete), axis=1) >= 0).all(), "More synapses used than available"
        self.C = C_discrete.astype(np.int)

    def bin_omega(self, C_real, min=-0.339, max=0.412):
        """
        Given: C_real:    Copy of the recurrent connection matrix from the simulation.
               min:       The min value of the recurrent matrix obtained from the simulations.
               max:       The max value of the recurrent matrix obtained from the simulations.
        Returns:          Discretized weight matrix.
        """
        
        np.fill_diagonal(C_real, 0)

        if(np.min(C_real) < min or np.max(C_real) > max):
            w = ("Recurrent matrix exceeds minimum or maximum. Max: %.3f, Min: %.3f" % (np.max(C_real),np.min(C_real)))
            warnings.warn(w, RuntimeWarning)
        
        """#threshold
        C_real[np.abs(C_new_real) < 0.05] = 0"""
        
        # All elements that are bigger than max will be set to max, same for min
        C_real[C_real > max] = max
        C_real[C_real < min] = min

        # Scale the new weights with respect to the range
        C_new_discrete = np.zeros(C_real.shape)
        
        max_syn = self.parameters["dynapse_maximal_synapse_o"]

        _, bin_edges = np.histogram(C_real.reshape((-1,1)), bins = 2*max_syn, range=(min,max))
        C_new_discrete = np.digitize(C_real.ravel(), bins = bin_edges, right = True).reshape(C_new_discrete.shape) - max_syn
        
        assert (C_new_discrete <= max_syn).all() and (C_new_discrete >= -max_syn).all(), "Error, have value > or < than max/min in Omega"
                    
        number_available_per_neuron = 62 - np.sum(np.abs(self.F), axis=1)

        if(not ((number_available_per_neuron - np.sum(np.abs(C_new_discrete), axis=1)) >= 0).all()):
            # Reduce the number of weights here, if necessary

            for idx in range(C_new_discrete.shape[0]):
                num_available = number_available_per_neuron[idx]
                num_used = np.sum(np.abs(C_new_discrete[idx,:]))

                # Use sorting + cutoff to keep the most dominant weights
                sorted_indices = np.flip(np.argsort(np.abs(C_new_discrete[idx,:])))
                sub_sum = 0; i = 0
                while(sub_sum < num_available):
                    if(i == len(sorted_indices)):
                        break
                    sub_sum += np.abs(C_new_discrete[idx,:])[sorted_indices[i]]
                    i += 1
                tmp = np.zeros(len(sorted_indices))
                tmp[sorted_indices[0:i-1]] = C_new_discrete[idx,sorted_indices[0:i-1]]
                C_new_discrete[idx,:] = tmp

                # Uses random subsampling to decide which weights to reduce
                """while(num_used > num_available):
                    ind_non_zero = np.nonzero(C_new_discrete[idx,:])[0]
                    rand_ind = np.random.choice(ind_non_zero, 1)[0]
                    if(C_new_discrete[idx,rand_ind] > 0):
                        C_new_discrete[idx,rand_ind] -= 1
                    else:
                        C_new_discrete[idx,rand_ind] += 1
                    num_used -= 1"""

        assert ((number_available_per_neuron - np.sum(np.abs(C_new_discrete), axis=1)) >= 0).all(), "More synapses used than available"

        if(self.debug):
            print("Number of neurons used: %d / %d" % (np.sum(np.abs(C_new_discrete)), np.sum(number_available_per_neuron)))

        self.C = C_new_discrete
        return C_new_discrete


