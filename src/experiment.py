
from extract import POLOCM
from traces import *
from observation import *
from convertor import TopoConvertor
from sklearn.metrics import f1_score 
from planner import PseudoPlanner
import json
import pandas as pd
import numpy as np



def experiment(file_name, dods, measurement, seed= None):
    file_path = '../data/' + file_name + '.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    headers = [
        'domain',
        'index',
        'type',
        'total_length',
        'size',
        'average_length',
        'measurement',
        'runtime',
        'f1_score',
        'executability'
    ]
    res = pd.DataFrame(headers=headers)
    for dod in dods:
        for learning_obj in data:
            domain = learning_obj['domain']
            index = learning_obj['index']
            type = learning_obj['type']
            total_length = learning_obj['total_length']
            raw_traces = learning_obj['traces']
            size = len(raw_traces)
            traces = []
            for raw_trace in raw_traces:
                steps = []
                for i, raw_step in enumerate(raw_trace):
                    action_name = raw_step['action']
                    obj_names = raw_step['obj_names']
                    objs = [PlanningObject('na', obj) for obj in obj_names]
                    action = Action(action_name, objs)
                    step = Step(None, action, i)
                    steps.append(step)
                trace = Trace(steps)
                traces.append(trace)
            tracelist = TraceList(traces)
            obs_tracelist = tracelist.tokenize(ActionObservation, ObservedTraceList)

            convertor = TopoConvertor(measurement, rand_seed=seed)
            po_tracelist = tracelist.topo(convertor, dod)
            obs_po_tracelist = po_tracelist.tokenize(PartialOrderedActionObservation, ObservedPartialOrderTraceList)
            runtime, f1_score, executability = single(
                obs_po_tracelist,
                obs_tracelist, 
                domain_filename=f"{domain}_{index}_{type}_tl{total_length}_size{size}_{measurement}"
            )
            row = {
                'domain': domain,
                'index': index,
                'type': type,
                'total_length': total_length,
                'size': size,
                'average_length': total_length/size,
                'measurement': measurement,
                'runtime': runtime,
                'f_score': f1_score,
                'executability':executability,
            }
            res= res.append(row, ignore_index=True)
    res.to_csv('../output/experiment/res/' + file_name + "_"+measurement + ".csv")


def single(obs_po_tracelist: TraceList,obs_tracelist, domain_filename="DOMAIN_0"):
    
    model, AP, runtime = POLOCM(obs_po_tracelist)
    sorts = POLOCM._get_sorts(obs_tracelist)
    AML, _ = POLOCM._locm2_step1(obs_tracelist, sorts)
    f1_score = get_AP_f1_score(AP, AML)

    file_path = "../output/experiment/pddl/" + domain_filename + ".pddl"
    model.to_pddl(domain_filename=file_path)
    # TODO: test also the learned order?  and the learned order to the ground domain
    executablilty = get_executability(obs_tracelist, file_path)

    return runtime, f1_score, executablilty


def get_AP_f1_score(AP, AML):
    if (len(AP)==0):
        return -1
    if (len(AP) != len(AML)):
        return -1
    res = []
    for sort, m1 in enumerate(AP):
        m1 = np.where(m1>0, 1, 0)
        l1 = m1.flatten()
        m2 = np.where(AML[sort]>0, 1,0)
        l2 = m2.flatten()
        res.append(f1_score(l1,l2))
    return sum(res)/len(res)

def get_executability(obs_tracelist, domain_filename):
    if (len(obs_tracelist)==0):
        return -1
    pp = PseudoPlanner(domain_filename)
    res = []
    for trace in obs_tracelist:
        actions = [step.action for step in trace]
        res.append(pp.check_executability(actions))

    return sum(res)/len(res)


