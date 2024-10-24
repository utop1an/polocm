
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
import random

DEBUG = False
SOLVER = "default"
ET:int = 2
CT:int = 4

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

def run_single_experiment(output_dir, dod, learning_obj, measurement, seed, verbose, logger):
    """Runs a single experiment given the necessary parameters."""

    domain = learning_obj['domain']
    index = learning_obj['index']
    total_length = learning_obj['total_length']
    raw_traces = learning_obj['traces']
    size = len(raw_traces)

    logger.info(f"Running {domain}-lo.{learning_obj['id']}-{dod} ...")

    
    try:
        traces = []
        for raw_trace in raw_traces:
            steps = []
            for i, raw_step in enumerate(raw_trace):
                action_name = raw_step['action']
                obj_names = raw_step['objs']
                objs = []
                for obj in obj_names:
                    obj_name, obj_type = obj.split("?")
                    objs.append(PlanningObject(obj_type, obj_name))
                action = Action(action_name, objs)
                step = Step(State(), action, i)
                steps.append(step)
            trace = Trace(steps)
            traces.append(trace)

        tracelist = TraceList(traces)
        obs_tracelist = tracelist.tokenize(ActionObservation, ObservedTraceList)
        actual_dod = 0
        error_rate = 0
        if dod == 0:
            runtime, accuracy_val, executability, result = single_locm2(
                obs_tracelist,
                domain_filename=f"{domain}_{index}_{learning_obj['id']}_dod{dod}",
                output_dir=output_dir,
                verbose=verbose,
                logger = logger
            )
            
        else:
            poats = learning_obj['pos']
            i = int((dod * 10)-1)
            poat = poats[i]
            actual_dod = poat['actual_dod']
            inds = poat['traces_inx']
            pos = poat['po']
            po_traces = []
            
            for i,trace in enumerate(tracelist):
                po = pos[i]
                ind = inds[i]
                po_steps =[]
                for j,po_step_ind in enumerate(ind):
                    ori_step = trace[po_step_ind]
                    po_step = PartialOrderedStep(ori_step.state, ori_step.action, ori_step.index, po[j])
                    po_steps.append(po_step)
                po_traces.append(PartialOrderedTrace(po_steps, actual_dod))
            
            po_tracelist = TraceList(po_traces)
            obs_po_tracelist = po_tracelist.tokenize(PartialOrderedActionObservation, ObservedPartialOrderTraceList)

            runtime, accuracy_val,error_rate, executability, result = single(
                obs_po_tracelist,
                obs_tracelist,
                domain_filename=f"{domain}_{index}_{learning_obj['id']}_dod{dod}",
                output_dir=output_dir,
                logger=logger,
                verbose=verbose
            )
        clear_output(output_dir)

    except GeneralTimeOut as t:
        runtime, accuracy_val, executability, result = (1200,0,0), 0, 0, f"Timeout"
    except Exception as e:
        runtime, accuracy_val, executability, result = (0, 0, 0), 0, 0, e
        logger.error(f"Error during experiment for domain {domain}: {e}")

    polocm_time, locm2_time, locm_time = runtime
    logger.info(f"{domain}-lo.{learning_obj['id']}-{dod}  DONE")

    result_data = {
        'lo_id': learning_obj['id'],
        'dod': dod,
        'actual_dod': actual_dod,
        'domain': domain,
        'index': index,
        'num_objects': learning_obj['number_of_objects'],
        'total_length': total_length,
        'size': size,
        'measurement': measurement,
        'runtime': sum(runtime),
        'polocm_time': polocm_time,
        'locm2_time': locm2_time,
        'locm_time': locm_time,
        'accuracy': accuracy_val,
        'error_rate': error_rate,
        'executability': executability,
        'result': result
    }
    return result_data




def write_result_to_csv(output_dir,dod, result_data, logger):
    """Writes the result data to a CSV file in a thread-safe manner."""
    csv_file_path = os.path.join(output_dir, f"results_{dod}.csv")
    with lock:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a') as csv_file:
            if not file_exists:
                headers = result_data.keys()
                csv_file.write(','.join(headers) + '\n')

            for data in result_data:

                values = [str(data[key]) for key in data.keys()]
                csv_file.write(','.join(values) + '\n')


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600, source="polocm")
def single(obs_po_tracelist ,obs_tracelist: ObservedTraceList, domain_filename, output_dir ,logger, verbose=False):
    try: 
        remark = []
        model, AP, runtime = POLOCM(obs_po_tracelist, solver_path=SOLVER, prob_type='polocm', cores=CT, logger=logger )
        filename = domain_filename + ".pddl"
        file_path = os.path.join(output_dir, "pddl", filename)
        tmp_file_path = os.path.join(output_dir, "pddl", "tmp", filename)
        model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)

        sorts = POLOCM._get_sorts(obs_tracelist)
        AML, _, __ = POLOCM._locm2_step1(obs_tracelist, sorts)
        accuracy_val,error_rate, r = get_AP_accuracy(AP, AML, verbose=verbose)
        if (r):
            remark.append(r)
        executabililty, r = get_executability(obs_tracelist, domain_filename=file_path)
        if r:
            remark.append(r)
        if len(remark)==0:
            remark = ['Success']
    except POLOCMTimeOut as t:
        return (1200,0,0), 0, 0,0, f"Timeout"
    except Exception as e:
        print(f"Error: {e}")
        return (0,0,0), 0,0, 0, e
    return runtime, accuracy_val,error_rate, executabililty, " ".join(remark)


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600, source="locm2")
def single_locm2(obs_tracelist: ObservedTraceList, domain_filename, output_dir, verbose=False, logger=None):
    try: 
        remark = []
        model,_, runtime = POLOCM(obs_tracelist, prob_type="locm2", logger=logger)
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
        return (0,1200,0), 0, 0, f"Timeout"
    except Exception as e:
        print(f"Error: {e}")
        return (0,0,0), 0, 0, e
    return runtime, 1, executabililty, " ".join(remark)

def get_AP_accuracy(AP, AML, verbose=False):
    if (len(AP)==0):
        return 0,1, "AP Empty"
    if (len(AP) != len(AML)):
        return 0,1, "AP Invalid Length"
    acc = []
    err = []
    for sort, m1 in AP.items():
        m1 = m1.reindex(index=AML[sort].index, columns=AML[sort].columns) 
        m1 = np.where(m1>0, 1, 0)
        l1 = m1.flatten()

        m2 = np.where(AML[sort]>0, 1,0)
        l2 = m2.flatten()
        if (verbose):
            print(f"sort{sort}-AP array [learned]: {l1}")
            print(f"sort{sort}-AP array [ground ]: {l2}")
        acc.append(sum(l1==l2)/len(l1))
        err.append(np.sum((l2==0)& (l1==1))/len(l1)) # false positive rate?
    return sum(acc)/len(acc), sum(err)/len(err), None

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


def experiment(input_filepath, output_dir, dod, measurement, seed=None, verbose=False):
    log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join("./logs", log_filename)
    logger = setup_logger(log_filepath)

    logger.info("Experiment Start...")
    logger.info(f"Using {ET} threads for parallel processing.")
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    if seed:
        logger.info(f"Setting seed to {seed}")
        random.seed(seed)

    tasks = []
    for learning_obj in data:
        tasks.append((output_dir, dod, learning_obj, measurement, seed, verbose, logger))
        
    
    if DEBUG:
        tasks = random.sample(tasks, 15)

    if ET > 1:
        logger.info("Running experiment in multiprocessing...")
        with Pool(processes=ET) as pool:
            res= pool.starmap_async(run_single_experiment, tasks).get()
            write_result_to_csv(output_dir, dod, res, logger)
    else:
        logger.info("Running experiment in sequential...")
        res= []
        for task in tasks:
            r= run_single_experiment(*task)
            res.append(r)
        write_result_to_csv(output_dir, dod, res, logger)
    logger.info("Experiment completed.")

def main(args):
    global SOLVER, DEBUG, ET, CT
    input_filepath = args.i
    output_dir = args.o
    seed = args.s
    experiment_threads = args.et
    cplex_threads = args.ct
    dod = args.d
    cplex_dir = args.cplex
    debug = args.debug
    if debug:
        DEBUG = True
    if experiment_threads:
        ET = experiment_threads
    if cplex_threads:
        CT = cplex_threads
    dods = [0, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
    if dod not in dods:
        print(f"Invalid dod {dod}. Choose from {dods}")
        return

    if ET < 1 or CT <1:
        print("Invalid number of threads. Choose a number greater than 0")
        return

    if not os.path.exists(cplex_dir):
        print("No cplex solver provided, defualt pulp solver will be used for MLP")
        SOLVER = "defualt"
    else:
        SOLVER = cplex_dir

    if not os.path.exists(input_filepath):
        print(f"Input file {input_filepath} does not exist")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+"/pddl"):
        os.makedirs(output_dir+"/pddl")
    if not os.path.exists(output_dir+"/pddl/tmp"):
        os.makedirs(output_dir+"/pddl/tmp")

    log_dir = ("./logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pddl_dir = os.path.join(output_dir, "pddl")
    if not os.path.exists(pddl_dir):
        os.makedirs(pddl_dir)


    
    experiment(input_filepath,output_dir, dod, 'flex', seed= seed,verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--i', type=str, help='Input trainning file name')
    parser.add_argument('--o', type=str, help='Output directory')
    parser.add_argument('--d', type=float, default=0, help='dod')
    parser.add_argument('--s', type=int, default=42, help='Rand seed')
    parser.add_argument('--et', type=int, default=2, help='Number of threads for experiment')
    parser.add_argument("--cplex", type=str, default="./", help="Path to cplex solver")
    parser.add_argument('--ct', type=int, default=4, help='Number of threads for cplex')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    main(args)