import sys, os

import helper_functions as hf

istest = len(sys.argv) == 3 and sys.argv[2] == 'test'

if not istest:
    print("Updating git")
    hf.git_it()

if len(sys.argv)>1:
    parameter_file = sys.argv[1]
else:
    parameter_file = 'params.py'
if istest:
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


if istest:
    params.update({"sumatra_label": 'test_sim'})
else:
    params.update({"sumatra_label": smt_record.label})

# p = {}

import time
start_time = time.time()

# Run simulation
params = hf.folders_setup(params, smt_record)
# hf.log_dir(p)
# print("hf.log_dir end")
# hf.redirect(p)
# print("hf.redirect end")


local_vars = {}
globas = {'params':params,
               'sys.argv':sys.argv}
if sys.version[0] == '2':
    execfile(params['model_file'],globas)
elif sys.version[0] == '3':
    exec(open(params['model_file']).read(),globas)
else:
    print("unknown python version!")
    sys.exit(1)

if not istest:
    smt_record.duration = time.time() - start_time
    smt_record.output_data = smt_record.datastore.find_new_data(smt_record.timestamp)
    project.add_record(smt_record)

    project.save()

sys.stdout.flush()
sys.stderr.flush()

