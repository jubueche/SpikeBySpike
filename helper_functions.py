import sys,os

def git_it():
    import git as gp
    repo = gp.repo.Repo('./')
    if repo.is_dirty():
        os.system('git add -u')
        # print('Insert a comment for this code (leave blank for default): ')
        print(sys.version)
        if sys.version[0] == '2':
            try:
                comment = raw_input('Insert a comment to git for this code (leave blank for default): ')
            except EOFError:
                comment = ''
            except Exception:
                comment = ''
        elif sys.version[0] == '3':
            try:
                comment = input('Insert a comment to git for this code (leave blank for default): ')
            except EOFError:
                comment = ''
            except Exception:
                comment = ''
        else:
            print('unknown python version')
            sys.exit(1)
        print("Updating git")
        if len(comment) == 0:
            os.system('git commit -m "WIP for Sumatra"')
        elif comment == 'B':
            os.system('git commit -m "Bugfix"')
        else:
            os.system('git commit -m "%s"'%comment)

def smt_it(parameter_file):
    import sumatra as smt
    from sumatra.projects import load_project
    from sumatra.parameters import build_parameters
    params = build_parameters(parameter_file)
    project = load_project()
    print(sys.version)
    if sys.version[0] == '2':
        try:
            reason = raw_input('Insert a reason to smt for running this simulation (leave blank for default): ')
        except EOFError:
            reason = ''
        except Exception:
            reason = ''
    elif sys.version[0] == '3':
        try:
            reason = input('Insert a reason to smt for running this simulation (leave blank for default): ')
        except EOFError:
            reason = ''
        except Exception:
            reason = ''
    else:
        print('unknown python version')
        sys.exit(1)
    if len(reason) == 0:
        reason = "Reason not provided. Test sim for reasons stated above (Maybe)."
    record = project.new_record(parameters=params,
                                reason=reason)
    return params, project, record

def folders_setup(p, params, rank=0, key='simConfig'):
    p[key].saveFolder = "Data/%s" % (params["sumatra_label"])
    filename = p[key].filename if p[key].filename else "sim_data"
    p[key].filename = "Data/%s/%s" % (params["sumatra_label"], filename)
    if rank == 0:
        os.system('mkdir -p %s'%p[key].saveFolder)
        with open('last_sim_dir.txt','w') as dirname:
            dirname.write(p[key].saveFolder+'/')
    return p

def log_dir(p, rank=0):
    if rank == 0:
        os.system('mkdir -p %s/%s'%(p['simConfig'].saveFolder,'logs'))

def redirect(p, rank=0):
    sys.stdout = sys.stderr = open('%s/%s/cpu_%i_run.log'%(p['simConfig'].saveFolder,'logs',rank),'w')
    #open('%s/%s/cpu_%i_run.log'%(p['simConfig'].saveFolder,'logs',rank),'w')
        
def figure_files(p, key='simConfig'):
    for an_type in ['plotTraces','plot2Dnet','plotRaster','plotSpikeHist','plotSpikeStats']:
        if an_type in p[key].analysis.keys():
            p[key].analysis[an_type]['saveFig'] = '%s/%s'%(p[key].saveFolder,
                                                           p[key].analysis[an_type]['saveFig'])

def figure_files_batch(p, key='simConfig', simLabel=None):
    print('Creating %s/%s'%(p[key].saveFolder,simLabel))
    os.system('mkdir -p %s/%s'%(p[key].saveFolder,simLabel))
    for an_type in ['plotTraces','plot2Dnet','plotRaster','plotSpikeHist','plotSpikeStats']:
        if an_type in p[key].analysis.keys():
            p[key].analysis[an_type]['saveFig'] = '%s/%s/%s'%(p[key].saveFolder,
                                                              simLabel,
                                                              p[key].analysis[an_type]['saveFig'])
    print ('Data folders created.')
