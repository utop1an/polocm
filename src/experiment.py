
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
import logging
import datetime

DEBUG = False

lock= Lock()

# Setup logger
def setup_logger(log_file):
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger

def run_single_experiment(args):
    """Runs a single experiment given the necessary parameters."""
    output_dir, dod, learning_obj, measurement, time_limit, seed, verbose, logger = args

    domain = learning_obj['domain']
    index = learning_obj['index']
    difficulty = learning_obj['difficulty']
    total_length = learning_obj['total_length']
    raw_traces = learning_obj['traces']
    size = len(raw_traces)

    logger.info(f"Running experiment for domain {domain} with DOD {dod}...")

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
        if dod == 0:
            runtime, accuracy_val, executability, result = single_locm2(
                obs_tracelist,
                domain_filename=f"{domain}_{difficulty}_{index}_tl{total_length}_size{size}_{measurement}_dod{dod}",
                output_dir=output_dir,
                time_limit=time_limit,
                verbose=verbose
            )
        else:
            convertor = TopoConvertor(measurement, strict=True, rand_seed=seed)
            po_tracelist, actual_dod = tracelist.topo(convertor, dod)
            obs_po_tracelist = po_tracelist.tokenize(PartialOrderedActionObservation, ObservedPartialOrderTraceList)

            logger.info(f"Running POLOCM for domain {domain}...")
            runtime, accuracy_val, executability, result = single(
                obs_po_tracelist,
                obs_tracelist,
                domain_filename=f"{domain}_{difficulty}_tl{total_length}_size{size}_{measurement}_dod{dod}",
                output_dir=output_dir,
                time_limit=time_limit,
                verbose=verbose
            )
    except GeneralTimeOut as t:
        runtime, accuracy_val, executability, result = tuple(i * 2 for i in time_limit), 0, 0, f"Timeout: {t}"
    except Exception as e:
        runtime, accuracy_val, executability, result = (0, 0, 0), 0, 0, e
        logger.error(f"Error during experiment for domain {domain}: {e}")

    polocm_time, locm2_time, locm_time = runtime
    logger.info(f"{domain}-{difficulty}-tl{total_length}-{dod}-> Runtime: {runtime}, Accuracy: {accuracy_val}, Executability: {executability}")

    clear_output(output_dir)

    result_data = {
        'lo_id': learning_obj['id'],
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

    write_result_to_csv(output_dir, result_data, logger)

    return result_data

def experiment(input_filepath, output_dir, dods, measurement, cores=1, time_limit=[600, 600, 300], seed=None, verbose=False):
    log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(output_dir, "logs", log_filename)
    logger = setup_logger(log_filepath)

    logger.info("Experiment Start...")
    
    logger.info(f"Reading data from {file_path}...")
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    if seed:
        logger.info(f"Setting seed to {seed}")
        random.seed(seed)

    task = []
    for dod in dods:
        for learning_obj in data:
            
            task.append((output_dir, dod, learning_obj, measurement, time_limit, seed, verbose, logger))
    
    if DEBUG:
        tasks = random.sample(task, 50)

    with Pool(processes=cores) as pool:
        pool.starmap_async(run_single_experiment, tasks).get()

    logger.info("Experiment completed.")

def write_result_to_csv(output_dir, result_data, logger):
    """Writes the result data to a CSV file in a thread-safe manner."""
    csv_file_path = os.path.join(output_dir, "results.csv")

    with lock:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a') as csv_file:
            if not file_exists:
                headers = result_data.keys()
                csv_file.write(','.join(headers) + '\n')

            values = [str(result_data[key]) for key in result_data.keys()]
            csv_file.write(','.join(values) + '\n')

    logger.info(f"Results written to {csv_file_path}")


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600)
def single(obs_po_tracelist: TraceList,obs_tracelist, domain_filename, output_dir, time_limit , verbose=False):
    try: 
        remark = []
        model, AP, runtime, mlp_info = POLOCM(obs_po_tracelist, time_limit=time_limit, solver_type='cplex', debug=debug)
        filename = domain_filename + ".pddl"
        file_path = os.path.join(output_dir, "pddl", filename)
        tmp_file_path = os.path.join(output_dir, "pddl", "tmp", filename)
        model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)

        sorts = POLOCM._get_sorts(obs_tracelist)
        AML, _, __ = POLOCM._locm2_step1(obs_tracelist, sorts)
        accuracy_val, r = get_AP_accuracy(AP, AML, verbose=verbose)
        if (r):
            remark.append(r)
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
def single_locm2(obs_tracelist: TraceList, domain_filename, output_dir, time_limit, verbose=False):
    try: 
        remark = []
        model, runtime = POLOCM(obs_tracelist, prob_type="locm2", time_limit=time_limit, debug=debug)
        filename = domain_filename + ".pddl"
        file_path = os.path.join(output_dir, "pddl", filename)
        tmp_file_path = os.path.join(output_dir, "pddl", "tmp", filename)
        model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)
      
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
def clear_output(output_dir):
    folder_path = os.path.join(output_dir, "pddl", "tmp")
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

def main(args):
    input_filepath = args.i
    output_dir = args.o
    seed = args.s
    cores = args.c
    measurement = args.m
    time_limit = args.l
    task_type = args.t

    if task_type not in ["polocm", "locm2"]:
        print("Invalid task type. Choose from polocm, locm2")
        return
    if task_type == "polocm":
        dods = [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
    else:
        dods = [0]

    if measurement not in ["flex", "width"]:
        print("Invalid measurement type. Choose from flex, width")
        return

    if cores < 1:
        print("Invalid number of cores. Choose a number greater than 0")
        return
    
    if cores > os.cpu_count():
        print(f"Number of cores {cores} is greater than available cores {os.cpu_count()}")
        return

    if not os.path.exists(input_filepath):
        print(f"Input file {input_filepath} does not exist")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pddl_dir = os.path.join(output_dir, "pddl")
    if not os.path.exists(pddl_dir):
        os.makedirs(pddl_dir)

    if time_limit and len(time_limit) > 3:
        print("Invalid time limit. Max length 3")
        return

    if len(time_limit) ==2:
        time_limit.append(300)
    elif len(time_limit) ==1:
        time_limit.append(600)
        time_limit.append(300)
    elif len(time_limit) ==0:
        time_limit = [600,600,300]

    
    experiment(input_filepath,output_dir, dods, measurement, cores= cores,time_limit=time_limit, seed= seed,verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--i', type=str, help='Input trainning file name')
    parser.add_argument('--o', type=str, help='Output directory')
    parser.add_argument('--m', type=str, default="flex", help='Measurement type')
    parser.add_argument('--s', type=int, default=42, help='Rand seed')
    parser.add_argument('--c', type=int, default=6, help='Number of cores')
    parser.add_argument('--t', type=string, default="polocm", help='Type of task, polocm or locm2')
    parser.add_argument('--l', type=int, nargs="+",default=[600,600,300], help='Time limit, max length 3, for [polocm, locm2, locm] respectively')

    args = parser.parse_args()
    main(args)