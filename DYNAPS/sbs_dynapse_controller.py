"""
Created on Thu Sep 26 22:11:04 2019

@author: dzenn
"""
import numpy as np
from random import sample
from time import sleep, time
import operator
import rpyc
import itertools

class SBSController():
    """
        Spike By Spike project Dynapse chip controller class
        
    """
    
    def __init__(self, start_neuron, chip_id, c, core_id, debug = False, base_isi=90, num_signals=2):
        
        self.start_neuron = start_neuron
        self.num_neurons = 20
        self.chip_id = chip_id
        self.core_id = core_id
        self.debug = debug
        self.c = c
        self.base_isi = base_isi
        
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
        
        self.population = [n for n in self.neurons if n.get_chip_id()==self.chip_id
             and n.get_core_id()==self.core_id
             and n.get_neuron_id() >= self.start_neuron
             and n.get_neuron_id() < self.start_neuron + self.num_neurons]
        self.population_ids = [n.get_chip_id()*1024 + n.get_core_id()*256 + n.get_neuron_id() for n in self.population]

        """self.dynapse.clear_cam(self.chip_id)
        if(self.debug):
            print("Finished clearing CAMs")"""
        
    @classmethod
    def from_default(self):
        c = rpyc.classic.connect("localhost", 1300)
        RPYC_TIMEOUT = 300 #defines a higher timeout
        c._config["sync_request_timeout"] = RPYC_TIMEOUT  # Set timeout to higher level

        return SBSController(start_neuron=1, chip_id=1, c=c, core_id=0, debug=True)
        
    def run_single_trial(self):
        pass
    
    
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

        
    def load_resources(self):
        """
            1. Loading all resource files: input weights and preloaded spiketrains.
            2. Injecting spikes of a dummy neuron to account for max. ISI.
            3. Converting to ISI's
            4. Generating FPGA events
            5. Loading Spike Generator
        """
        
        self.spikes_up = []
        self.spikes_down = []

        for i in range(self.num_signals):
                    self.spikes_up.append(np.load(("Resources/x%d_up.dat" % i), allow_pickle=True))
                    self.spikes_down.append(np.load(("Resources/x%d_down.dat" % i), allow_pickle=True))
                

        # conn_up[i] corresponds to the weight of the connections from neuron i to all 20 neurons
        self.conn_up = []
        self.conn_down = []

        for i in range(self.num_signals):
                    self.conn_up.append(np.load(("Resources/conn_x%d_up.dat" % i), allow_pickle=True))
                    self.conn_down.append(np.load(("Resources/conn_x%d_down.dat" % i), allow_pickle=True))
        

        spike_times = self.compile_preloaded_stimulus(dummy_neuron_id = 255)
        
        # Convert to ISI
        (signal_isi, neuron_ids) = self.spikes_to_isi(spike_times=spike_times[:,1], neurons_id=spike_times[:,0], use_microseconds=False)

        # Get the FPGA events
        fpga_events = self.get_fpga_events(signal_isi, neuron_ids)
        
        # Load spike gen
        self.load_spike_gen(fpga_events=fpga_events, isi_base=self.base_isi, repeat_mode=False)

        # Create the feedforward connections based on the matrices
        weights = [self.conn_up, self.conn_down]
        weights = [elem.astype(int).tolist() for w in weights for elem in w] # Create simple list of lists and cast to int
        
        vn_list = []; pop_neur_list = []
        # For every input neuron        
        for current_vn_idx,weight in enumerate(weights):
            for current_pop_neuron_idx,w in enumerate(weight):
                vn_tmp = [self.v_neurons[current_vn_idx] for _ in range(w)]
                pop_n_tmp = [self.population[current_pop_neuron_idx] for _ in range(w)]
                vn_list.append(vn_tmp)
                pop_neur_list.append(pop_n_tmp)

        vn_list = [elem for l in vn_list for elem in l]
        pop_neur_list = [elem for l in pop_neur_list for elem in l]

        """self.connector.add_connection_from_list(vn_list, pop_neur_list, self.SynTypes.SLOW_EXC)
        self.model.apply_diff_state()"""
        
        
    
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
    
    def set_connections_F(self):
        
        # Number of neurons: 2*number of signals
        for w in self.conn_down:
            print(w)
        
    def plot_raster(self):
        pass
