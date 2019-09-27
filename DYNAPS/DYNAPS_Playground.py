import CtxDynapse
from NeuronNeuronConnector import DynapseConnector


model = CtxDynapse.model
neurons = model.get_shadow_state_neurons()
virtual_model = CtxDynapse.VirtualModel()
virtual_neurons = virtual_model.get_neurons()

spikegen = model.get_fpga_modules()[1]
connector = DynapseConnector()

# Clearing cams...
for chip_id in range(4):
    CtxDynapse.dynapse.clear_cam(chip_id)

core_id = 0
neuron_id = 140

connector.add_connection(pre=virtual_neurons[1], post=neurons[ neuron_id +
    256 * (core_id % 4)], synapse_type=CtxDynapse.DynapseCamType.FAST_EXC)
model.apply_diff_state()
isi_base = 900
unit_mult = isi_base / 90
rate = 20

# Setting inputs
fpga_event = CtxDynapse.FpgaSpikeEvent()
fpga_event.core_mask = 15  # Receiving cores
fpga_event.target_chip = 0  # Receiving chip
fpga_event.neuron_id = 1  # This needs to be the id of the virtual, i.e. sending, neuron

fpga_event.isi = int(((rate * 1e-6)**(-1)) / unit_mult)

# Loading fpga events...
spikegen.set_variable_isi(False)
spikegen.preload_stimulus([fpga_event])
spikegen.set_isi(int(((rate * 1e-6)**(-1)) / unit_mult))

spikegen.set_isi_multiplier(isi_base)
spikegen.set_repeat_mode(True)

spikegen.start()

# Stimulation starts!
CtxDynapse.dynapse.monitor_neuron(int(core_id / 4), neuron_id + 256 * (core_id % 4))