import sys, os
from subprocess import call
import helper_functions as hf
import json
import time

if(sys.argv[1] != None):
    parameter_file = sys.argv[1]
else:
    raise SystemExit('Error: Please add training parameters.')

is_test = False
if(len(sys.argv)>2):
    is_test = sys.argv[2] == "test"

if not is_test:
    print("Updating git")
    hf.git_it()

if is_test:
    params = {}
    global_vars = {}
    if sys.version[0] == '2':
        execfile(parameter_file, global_vars, params)
    elif sys.version[0] == '3':
        exec(open(parameter_file).read(), global_vars, params)
    else:
        print("unknown python version!")
        sys.exit(1)
    project = None
    smt_record = None
else:
    params, project, smt_record = hf.smt_it(parameter_file)


if is_test:
    params.update({"sumatra_label": 'test_sim'})
else:
    params.update({"sumatra_label": smt_record.label})


start_time = time.time()
params = hf.folders_setup(params, smt_record)
# Run simulation
call([sys.executable, 'spike_by_spike_training.py', params['saveFolder'], parameter_file])


if not is_test:
    smt_record.duration = time.time() - start_time
    smt_record.output_data = smt_record.datastore.find_new_data(smt_record.timestamp)
    project.add_record(smt_record)

    project.save()

sys.stdout.flush()
sys.stderr.flush()

