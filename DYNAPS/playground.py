"""
Before running this cell:
1) $ cd ~/Documents/NCS/cortexcontrol
2) $ sudo ./cortexcontrol
3) $ exec(open("./start_rpyc.py").read())
"""
from sbs_dynapse_controller import SBSController

sbs = SBSController.from_default()
sbs.load_resources()


# Connect 4 virtual neurons to 4 real neurons
sbs.connector.add_connection(sbs.v_neurons[1], sbs.neurons[1 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
sbs.connector.add_connection(sbs.v_neurons[2], sbs.neurons[2 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
sbs.connector.add_connection(sbs.v_neurons[3], sbs.neurons[3 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
sbs.connector.add_connection(sbs.v_neurons[4], sbs.neurons[4 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)

sbs.model.apply_diff_state()

sbs.spikegen.start()