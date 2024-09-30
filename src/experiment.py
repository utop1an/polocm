
from extract import POLOCM
from traces import *
from observation import *
from convertor import TopoConvertor
from sklearn.metrics import f1_score
from planner import PseudoPlanner
import json
import pandas as pd
import numpy as np
from utils import set_timer_throw_exc, GeneralTimeOut, POLOCMTimeOut
from utils.common_errors import InvalidModel, InvalidActionSequence, InvalidMLPTask
from multiprocessing import Pool, Lock
import os
import argparse

debug = {}
debug_domains = ['tidybot']
DEBUG = False

lock= Lock()

def run_single_experiment(args):
    """Runs a single experiment given the necessary parameters."""
    file_name, dod, learning_obj, measurement, time_limit, seed, verbose = args

    domain = learning_obj['domain']
    index = learning_obj['index']
    difficulty = learning_obj['difficulty']
    total_length = learning_obj['total_length']
    raw_traces = learning_obj['traces']
    size = len(raw_traces)

    print(f"Running for {domain} with DOD {dod}...")
    # Formatting raw traces
    traces = []
    for raw_trace in raw_traces:
        steps = []
        for i, raw_step in enumerate(raw_trace):
            action_name = raw_step['action']
            obj_names = raw_step['objs']
            objs = [PlanningObject('na', obj) for obj in obj_names]
            action = Action(action_name, objs)
            step = Step({}, action, i)
            steps.append(step)
        trace = Trace(steps)
        traces.append(trace)
    
    tracelist = TraceList(traces)
    obs_tracelist = tracelist.tokenize(ActionObservation, ObservedTraceList)

    try:
        if (dod ==0):
            runtime, accuracy_val, executability, result = single_locm2(
                obs_tracelist,
                domain_filename=f"{domain}_{difficulty}_{index}_tl{total_length}_size{size}_{measurement}_dod{dod}",
                time_limit=time_limit,
                verbose=verbose
            )
        else:
            # Converting partial ordered traces
            convertor = TopoConvertor(measurement, strict=True, rand_seed=seed)
            po_tracelist, actual_dod = tracelist.topo(convertor, dod)
            obs_po_tracelist = po_tracelist.tokenize(PartialOrderedActionObservation, ObservedPartialOrderTraceList)

            print("Running POLOCM...")
            runtime, accuracy_val, executability, result = single(
                obs_po_tracelist,
                obs_tracelist,
                domain_filename=f"{domain}_{difficulty}_tl{total_length}_size{size}_{measurement}_dod{dod}",
                time_limit=time_limit,
                verbose=verbose
            )
    except GeneralTimeOut as t:
        runtime, accuracy_val, executability, result = tuple(i*2 for i in time_limit), 0, 0, f"Timeout: {t}"
    except Exception as e:
        runtime, accuracy_val, executability, result = (0,0,0), 0, 0, e
   
    polocm_time, locm2_time, locm_time = runtime
    print(f"Finished experiment for {domain}. Runtime: {runtime}, F1 Score: {accuracy_val}, Executability: {executability}")

    clear_output()
    
    result_data = {
        'dod': dod,
        'domain': domain,
        'index': index,
        'total_length': total_length,
        'size': size,
        'difficulty': difficulty,
        'measurement': measurement,
        'runtime': sum(runtime),
        'polocm_time': polocm_time,
        'locm2_time': locm2_time,
        'locm_time': locm_time,
        'accuracy': accuracy_val,
        'executability': executability,
        'result': result
    }

    # Write result to CSV immediately after computing
    write_result_to_csv(file_name, result_data)
    
    return result_data

def experiment(file_name, dods, measurement, cores=1, time_limit=(600, 600, 300), seed=None, verbose=False):
    print("Experiment Start...")
    file_path = '../data/json_traces/' + file_name + '.json'
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    print("Reading data...")
    with open(file_path, 'r') as file:
        data = json.load(file)

    if seed:
        print(f"Setting seed to {seed}")

    # Prepare tasks for parallel processing
    tasks = []
    for dod in dods:
        for learning_obj in data:
            if DEBUG and learning_obj['domain'] not in debug_domains:
                continue
            tasks.append((file_name, dod, learning_obj, measurement, time_limit, seed, verbose))

    # Use Pool to process tasks in parallel
    with Pool(processes=cores) as pool:
        pool.map(run_single_experiment, tasks)

    print("Experiment completed.")


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600)
def single(obs_po_tracelist: TraceList,obs_tracelist, domain_filename, time_limit , verbose=False):
    try: 
        remark = []
        model, AP, runtime, mlp_info = POLOCM(obs_po_tracelist, time_limit=time_limit, solver_type='gurobi', debug=debug)
        file_path = "../output/experiment/pddl/" + domain_filename + ".pddl"
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        tmp_file_path = "../output/experiment/pddl/tmp/" + domain_filename + ".pddl"
        tmp_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), tmp_file_path))
        model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)

        print("Evaluating accuracy...")
        sorts = POLOCM._get_sorts(obs_tracelist)
        AML, _, __ = POLOCM._locm2_step1(obs_tracelist, sorts)
        accuracy_val, r = get_AP_accuracy(AP, AML, verbose=verbose)
        if (r):
            remark.append(r)
        print("Evaluating executability...")
        executabililty, r = get_executability(obs_tracelist, domain_filename=file_path)
        
        if r:
            remark.append(r)
        if len(remark)==0:
            remark = ['Success']
    except POLOCMTimeOut as t:
        return tuple(i*2 for i in time_limit), 0, 0, f"Timeout: {t}"
    except Exception as e:
        print(f"Error: {e}")
        return (0,0,0), 0, 0, e
    return runtime, accuracy_val, executabililty, " ".join(remark)


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600)
def single_locm2(obs_tracelist: TraceList, domain_filename, time_limit, verbose=False):
    try: 
        remark = []
        model, runtime = POLOCM(obs_tracelist, prob_type="locm2", time_limit=time_limit, debug=debug)
        file_path = "../output/experiment/pddl/" + domain_filename + ".pddl"
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
        tmp_file_path = "../output/experiment/pddl/tmp/" + domain_filename + ".pddl"
        tmp_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), tmp_file_path))
        model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)
      
        print("Evaluating executability...")
        executabililty, r = get_executability(obs_tracelist, domain_filename=file_path)
        if r:
            remark.append(r)
        if len(remark)==0:
            remark = ['Success']
    except POLOCMTimeOut as t:
        return tuple(i*2 for i in time_limit), 0, 0, f"Timeout: {t}"
    except Exception as e:
        print(f"Error: {e}")
        return (0,0,0), 0, 0, e
    return runtime, 1, executabililty, " ".join(remark)

def get_AP_accuracy(AP, AML, verbose=False):
    if (len(AP)==0):
        return 0, "AP Empty"
    if (len(AP) != len(AML)):
        return 0, "AP Invalid Length"
    res = []
    for sort, m1 in AP.items():
        m1 = m1.reindex(index=AML[sort].index, columns=AML[sort].columns) 
        m1 = np.where(m1>0, 1, 0)
        l1 = m1.flatten()

        m2 = np.where(AML[sort]>0, 1,0)
        l2 = m2.flatten()
        if (verbose):
            print(f"sort{sort}-AP array [learned]: {l1}")
            print(f"sort{sort}-AP array [ground ]: {l2}")
        res.append(sum(l1==l2)/len(l1))
    return sum(res)/len(res), None

# not used
# f1 score is not suitable for this task
def get_AP_f1_score(AP, AML, verbose=False):
    if (len(AP)==0):
        return 0, "Empty AP"
    if (len(AP) != len(AML)):
        return 0, "Different length of AP and AML"
    res = []
    for sort, m1 in AP.items():
        m1 = m1.reindex(index=AML[sort].index, columns=AML[sort].columns)
        m1 = np.where(m1>0, 1, 0)
        l1 = m1.flatten()

        m2 = np.where(AML[sort]>0, 1,0)
        l2 = m2.flatten()
        if (verbose):
            print(f"sort{sort}-AP array [learned]: {l1}")
            print(f"sort{sort}-AP array [ground ]: {l2}")
        res.append(f1_score(l2,l1, zero_division=0))
    return sum(res)/len(res), None

def get_executability(obs_tracelist, domain_filename=None, planner=None):
    if (planner and not domain_filename):
        pp = planner
        
    elif(domain_filename and not planner):
        pp = PseudoPlanner(domain_filename)
    else:
        raise Exception("Either domain_filename or planner should be provided")
    res = []
    for trace in obs_tracelist:
        actions = [step.action for step in trace]
        try: 
            res.append(pp.check_executability(actions))
        except Exception as e:
            return 0, str(e)

    return sum(res)/len(res), None


import shutil
# remove redundant task.pddl files autogenerated by the model
def clear_output():
    folder_path = "../output/experiment/pddl/tmp"
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder_path))
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate over the contents of the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Remove file if it is a file
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            # Remove directory if it is a directory
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted folder: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def write_result_to_csv(file_name, result_data):
    """Writes the result data to a CSV file in a thread-safe manner."""
    csv_file_path = f"../output/experiment/res/{file_name}.csv"
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), csv_file_path))
    with lock:
        # Append result_data to the CSV file
        # Implement the code to write to CSV
        with open(csv_file_path, 'a') as csv_file:
            # Assuming result_data is a dictionary and you want to write keys as headers
            # and values as the row
            if os.stat(csv_file_path).st_size == 0:
                # Write headers if file is empty
                headers = result_data.keys()
                csv_file.write(','.join(headers) + '\n')
            # Write the data row
            values = [str(result_data[key]) for key in result_data.keys()]
            csv_file.write(','.join(values) + '\n')

if __name__ == "__main__":
    seed = 42
    filename = 'plan_diff_s42_r1'
    dods = [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
    # dods = [0]
    measurement = 'flex'
    cores= 6
    experiment(filename, dods, measurement, cores= cores,time_limit=(600, 600, 300), seed= seed,verbose=False)