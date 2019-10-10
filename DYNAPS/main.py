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

#sbs.c.execute("exec(open('" + self_path + "/Resources/dynapse_biases.py').read())")
sbs.c.execute("exec(open('" + self_path + "/Resources/working_5.py').read())")

sbs.model.apply_diff_state()

sbs.run_single_trial(plot_raster=True)