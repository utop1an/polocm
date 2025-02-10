
from extract import POLOCMBASELINE
from traces import *
from observation import *
from sklearn.metrics import f1_score
from planner import PseudoPlanner
import json
import numpy as np
from utils import set_timer_throw_exc, GeneralTimeOut, POLOCMTimeOut
from multiprocessing import Pool, Lock
import os
import argparse
import logging
import datetime
import random

DEBUG = False

DATA = []

lock= Lock()


def read_json_file():
    json_file_path = "../data/traces_plan_r1_no_obj_lim.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return data

def run_single_experiment(learning_obj,dir, method,debug):
    print("Running experiment for learning object {0}, {1}".format(learning_obj['id'], learning_obj['domain']))
    domain = learning_obj['domain']
    lo_id = learning_obj['id']

    traces = []
    domain_los = [lo for lo in DATA if (lo['id'] != lo_id and lo['domain'] == domain)]
    candidates = random.sample(domain_los, max(10, len(domain_los)))
    for lo in candidates:
        obs_tracelist = lo['obs_tracelist']
        traces.extend(obs_tracelist)
    
    dods = [0, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
    for dod in dods:
        if (dod == 0):
            dod_str = "0.0"
        elif (dod == 1):
            dod_str = "1.0"
        else:
            dod_str = str(dod)
        domain_filename = dir + domain + "_0_" + str(lo_id) + "_dod" + dod_str + ".pddl"
        if not os.path.exists(domain_filename):
            print("File {0} does not exist".format(domain_filename))
            write_result_to_csv([lo_id, dod, domain,method, -1, -1])
            continue
        gt_domain_filename = "../data/goose-benchmarks/tasks/{0}/domain.pddl".format(domain)
        try:
            pp = PseudoPlanner(domain_filename, 'twoway', gt_domain_filename)
            e1s = []
            e2s = []
            for trace in traces:
                actions = [step.action for step in trace]
                e1, e2 = pp.check_executability(actions)
                e1s.append(e1)
                e2s.append(e2)
            exe1 = sum(e1s)/len(e1s)
            exe2 =  sum(e2s)/len(e2s)
    
            write_result_to_csv([lo_id, dod, domain,method, exe1, exe2])

        except Exception as e:
            print("Error in domain {0} with dod {1}: {2}".format(domain, dod, e))
            write_result_to_csv([lo_id, dod, domain,method, -1, -1])



def write_result_to_csv( result_data):
    """Writes the result data to a CSV file in a thread-safe manner."""
    csv_file_path = "../experiments/twoway_exe.csv"
    with lock:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a') as csv_file:
            if not file_exists:
                headers = ['lo_id', 'dod', 'domain', 'exe1', 'exe2']
                csv_file.write(','.join(headers) + '\n')
            values = [str(value) for value in result_data]
            csv_file.write(','.join(values) + '\n')





def experiment():
    global DATA
    data = read_json_file()

    tasks = []
    methods = [ "polocm"]

    for learning_obj in data:
        raw_traces = learning_obj['traces']
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
        learning_obj['obs_tracelist'] = obs_tracelist
        DATA.append(learning_obj)

    for method in methods:
        dir =  "../experiments/{m}/pddl/".format(m=method)
        for lo in random.choices(data, k= 10):
            tasks.append((lo,dir,method, None))  
        # for lo in data:
            # tasks.append((lo,dir,method, None))        

   
    with Pool(processes=8) as pool:
        pool.starmap_async(run_single_experiment, tasks).get()
            



if __name__ == "__main__":
    experiment()