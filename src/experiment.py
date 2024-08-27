
from extract import POLOCM
from traces import *
from observation import *
from convertor import TopoConvertor
from sklearn.metrics import f1_score 
from planner import PseudoPlanner
import json
import pandas as pd
import numpy as np



def experiment(file_name, dods, measurement, seed= None, verbose=False):
    print("Experiment Start...")
    file_path = '../data/json_traces/' + file_name + '.json'
    print("Reading data...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("done.")
    headers = [
        'dod',
        'domain',
        'index',
        'type',
        'total_length',
        'size',
        'average_length',
        'measurement',
        'runtime',
        'polocm_time',
        'locm2_time',
        'locm_time',
        'f1_score',
        'executability',
        'executability_ground_truth',
        'executability_diff'
    ]
    res = pd.DataFrame(columns=headers)
    print("Preparing ground truth data...")
    domains = {obj['domain'] for obj in data}
    planners_ground_truth = {domain: PseudoPlanner(f"../data/ground/{domain}.pddl") for domain in domains}
    print("done.")
    for dod in dods:
        print(f"dod of {dod}...")
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
                    obj_names = raw_step['objects']
                    objs = [PlanningObject('na', obj) for obj in obj_names]
                    action = Action(action_name, objs)
                    step = Step({}, action, i)
                    steps.append(step)
                trace = Trace(steps)
                traces.append(trace)
            tracelist = TraceList(traces)
            obs_tracelist = tracelist.tokenize(ActionObservation, ObservedTraceList)

            convertor = TopoConvertor(measurement,strict=True, rand_seed=seed)
            po_tracelist = tracelist.topo(convertor, dod)
            obs_po_tracelist = po_tracelist.tokenize(PartialOrderedActionObservation, ObservedPartialOrderTraceList)
            runtime, f1_score, executability, executability_ground_truth, executability_diff = single(
                obs_po_tracelist,
                obs_tracelist, 
                planners_ground_truth[domain],
                domain_filename=f"{domain}_{index}_{type}_tl{total_length}_size{size}_{measurement}_dod{dod}",
                verbose= verbose
            )
            polocm_time, locm2_time, locm_time = runtime
            print(f"{domain}-{index}[{type}, avg-len{total_length/size}]\n\t runtime: {runtime}, f1_score: {f1_score}, e: {executability}, e_gt: {executability_ground_truth}")
            row = {
                'dod': dod,
                'domain': domain,
                'index': index,
                'type': type,
                'total_length': total_length,
                'size': size,
                'average_length': total_length/size,
                'measurement': measurement,
                'runtime': sum(runtime),
                'polocm_time': polocm_time,
                'locm2_time': locm2_time,
                'locm_time': locm_time,
                'f1_score': f1_score,
                'executability':executability,
                'executability_ground_truth': executability_ground_truth,
                'executability_diff': executability_diff
            }
            res.loc[len(res)] = row
    res.to_csv('../output/experiment/res/' + file_name + "_"+measurement + ".csv")


def single(obs_po_tracelist: TraceList,obs_tracelist,planner_ground_truth, domain_filename, verbose=False):
    
    model, AP, runtime = POLOCM(obs_po_tracelist)
    

    file_path = "../output/experiment/pddl/" + domain_filename + ".pddl"
    tmp_file_path = "../output/experiment/pddl/tmp/" + domain_filename + ".pddl"
    model.to_pddl(domain_filename, domain_filename=file_path, problem_filename=tmp_file_path)

    sorts = POLOCM._get_sorts(obs_tracelist)
    AML, _ = POLOCM._locm2_step1(obs_tracelist, sorts)
    f1_score = get_AP_f1_score(AP, AML, verbose=verbose)
    
    executabililty = get_executability(obs_tracelist, domain_filename=file_path)
    executabililty_ground_truth = get_executability(obs_tracelist, planner=planner_ground_truth) 
    return runtime, f1_score, executabililty, executabililty_ground_truth, executabililty - executabililty_ground_truth


def get_AP_f1_score(AP, AML, verbose=False):
   
    if (len(AP)==0):
        return -1
    if (len(AP) != len(AML)):
        return -1
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
        res.append(f1_score(l2,l1))
    return sum(res)/len(res)

def get_executability(obs_tracelist, domain_filename=None, planner=None):
    assert len(obs_tracelist)!=0, "obs_tracelist should not be empty"
    if (planner and not domain_filename):
        pp = planner
    elif(domain_filename and not planner):
        pp = PseudoPlanner(domain_filename)
    else:
        assert False, "either planner or domain_filename should be provided"
    res = []
    for trace in obs_tracelist:
        actions = [step.action for step in trace]
        res.append(pp.check_executability(actions))

    return sum(res)/len(res)


