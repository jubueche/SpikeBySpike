"""
Before running this cell:
1) $ cd ~/Documents/NCS/cortexcontrol
2) $ sudo ./cortexcontrol
3) $ exec(open("./start_rpyc.py").read())
"""
import sbs_dynapse_controller

self_path = sbs_dynapse_controller.__file__
self_path = self_path[0:self_path.rfind('/')]

sbs = sbs_dynapse_controller.SBSController.from_default()
sbs.load_resources()

print(self_path)

sbs.c.execute("exec(open('" + self_path + "/Resources/dynapse_biases.py').read())")

# Connect 4 virtual neurons to 4 real neurons
#sbs.connector.add_connection(sbs.v_neurons[1], sbs.neurons[1 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
#sbs.connector.add_connection(sbs.v_neurons[2], sbs.neurons[2 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
#sbs.connector.add_connection(sbs.v_neurons[3], sbs.neurons[3 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)
#sbs.connector.add_connection(sbs.v_neurons[4], sbs.neurons[4 + 256 *(sbs.core_id % 4) + 1024*sbs.chip_id], sbs.SynTypes.SLOW_EXC)

sbs.model.apply_diff_state()

sbs.run_single_trial(plot_raster=True)