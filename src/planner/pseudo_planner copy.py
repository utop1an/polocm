
from collections import defaultdict
from math import e
import random

from matplotlib.pylab import rand
from requests import get
from traitlets import default
import pddl_parser
from traces.action import Action
from traces.fluent import PlanningObject
import translate.normalize as normalize
import os
from utils.common_errors import InvalidModel, InvalidActionSequence
from utils import (
    set_timer_throw_exc,
    GeneralTimeOut
)

random.seed(42)

class PseudoPlanner:
    def __init__(self, domain_filename, executability_type='overall', gt_domain_filename=None) -> None:
        self.domain_filename = domain_filename
        self.gt_domain_filename = gt_domain_filename
        try:
            self.initialize_task()
            self.executability_type = executability_type

        except Exception as e:
            raise Exception(f"Error parsing file{domain_filename}: {e} ")

    @set_timer_throw_exc(num_seconds=30, exception=GeneralTimeOut, max_time=30, source="pseudo planner")
    def initialize_task(self):
        """Initialize the task from domain and problem files. This should be called in the worker process."""
        domain = pddl_parser.open(self.domain_filename)
        normalize.normalize(domain)
        self.domain = domain

        if self.gt_domain_filename:
            gt_domain = pddl_parser.open(self.gt_domain_filename)
            normalize.normalize(gt_domain)
            self.gt_domain = gt_domain

       

    def check_executability(self,action_sequence, debug=False):
        if (self.executability_type == 'overall'):
            return self.get_overall_executability(action_sequence, debug)
        elif (self.executability_type == 'first_fail'):
            return self.get_first_fail_executability(action_sequence, debug)
        elif (self.executability_type == 'twoway'):
            return self.get_twoway_executabilities(action_sequence, debug)
        else:
            return self.get_overall_executability(action_sequence, debug)
        
    def get_twoway_executabilities(self,action_sequence, debug=False):
        if not self.domain:
            raise Exception("Domain not initialized")
        if not self.gt_domain_filename:
            raise Exception("GT Domain not given")
        
        if (len(action_sequence)==0):
            raise Exception("Error checking executability: Length 0")
        true_effs = set()
        type_objs = defaultdict(set)

        # as we don't know the init states, we can only record the visited effects
        # we assume unvisited effects are true since all given action seqs are valid
        visited = set()
        error_count = 0
        for i, a in enumerate(action_sequence):
            action = self.domain.get_action(a.name)
            
            # if action not found, meaning it has not been learned properly, with no precond or effects
            # we skip it, and add error count by 1
            if not action:
                error_count += 1
                continue
            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            objects_by_type = dict(zip(params, param_types))
           
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)

            for (name, type) in objects_by_type.items():
                type_objs[type].add(name)


            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)
            # not applicable
            if(len(invalid)>0):
                error_count += 1
                if debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)

            # apply action
            adds = set(e for _,e in op.add_effects)
            dels = set(e for _,e in op.del_effects)
            
            # mark visited effects
            visited = visited.union(adds).union(dels)

            true_effs = true_effs.union(adds)
            true_effs.difference_update(dels)

        exe_on_learned = 1-error_count/len(action_sequence)

        new_action_sequences = self.generate_new_action_sequence(type_objs, set(), set(), len(action_sequence), debug)
        exe_on_gt = self.get_gt_executability(new_action_sequences, debug)
        
        return exe_on_learned, exe_on_gt

    def generate_new_action_sequence(self,type_objs, init_effs, init_visited, length, debug=False):
        grounded_actions = self.domain.get_grounded_actions(type_objs)
        if debug:
            print("number of grounded actions:", len(grounded_actions))
            print("Grounded actions:", grounded_actions)
        true_effs = init_effs.copy()
        visited = init_visited.copy()
        def get_applicable_actions(_effs, _visited):
            applicable_actions = []
            for action in grounded_actions:
                preconditions = set(action.precondition)
                invalid = preconditions.difference(_effs)
                invalid = invalid.intersection(_visited)
                if (len(invalid)==0):
                    applicable_actions.append(action)
            return applicable_actions
        
        
        plans = []
        for _ in range(5):
            plan = []
            for i in range(length):
                candiates = get_applicable_actions(true_effs, visited)
                if (len(candiates) == 0):
                    break
                action = random.choice(candiates)

                plan.append(action)

                adds = set(e for _,e in action.add_effects)
                dels = set(e for _,e in action.del_effects)
                
                # mark visited effects
                visited = visited.union(adds).union(dels)

                true_effs = true_effs.union(adds)
                true_effs.difference_update(dels)
            plans.append(plan)
        action_seqs = []
        if debug:
            print("New action sequence:", plan)
        for plan in plans:
            action_seq = []
            for op in plan:
                a = op.name.strip().strip("()").split(" ")
                action_name = a[0]
                args = a[1:]

                params = [PlanningObject("na", obj) for obj in args if obj != 'zero']
                action = Action(action_name, params)
                action_seq.append(action)
            action_seqs.append(action_seq)
        if debug:
            print(action_seq)
        return action_seqs

    
    def get_gt_executability(self,action_sequences, debug=False):
        if (len(action_sequences)==0):
            print("cant find act seqs, exe 0")
            return 0
        res = []
        gt_planner = PseudoPlanner(self.gt_domain_filename, executability_type='overall')
        for act_seq in action_sequences:
            if (len(act_seq)==0):
                print("cant find valid act seq, exe 0")
                res.append(0)
                continue

            exe = gt_planner.check_executability(act_seq, debug)
            res.append(exe)
        return sum(res)/len(res)

    def get_overall_executability(self,action_sequence, debug=False):
        if not self.domain:
            raise Exception("Domain not initialized")
        if (len(action_sequence)==0):
            raise Exception("Error checking executability: Length 0")
        true_effs = set()
        # as we don't know the init states, we can only record the visited effects
        # we assume unvisited effects are true since all given action seqs are valid
        visited = set()
        error_count = 0
        for i, a in enumerate(action_sequence):
            action = self.domain.get_action(a.name)
            # if action not found, meaning it has not been learned properly, with no precond or effects
            # we skip it, and add error count by 1
            if not action:
                error_count += 1
                continue
            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)
            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)
            # not applicable
            if(len(invalid)>0):
                error_count += 1
                if debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)

            # apply action
            adds = set(e for _,e in op.add_effects)
            dels = set(e for _,e in op.del_effects)
            
            # mark visited effects
            visited = visited.union(adds).union(dels);

            true_effs = true_effs.union(adds)
            true_effs.difference_update(dels)

            if debug:
                print(f"action {op} executed")
                
        return 1-error_count/len(action_sequence)
        
    def get_first_fail_executability(self,action_sequence, debug=False):
        if not self.domain:
            raise Exception("Domain not initialized")
        if (len(action_sequence)==0):
            raise Exception("Error checking executability: Length 0")
        true_effs = set()

        # as we don't know the init states, we can only record the visited effects
        # we assume unvisited effects are true since all given action seqs are valid
        visited = set()
        for i, a in enumerate(action_sequence):
            action = self.domain.get_action(a.name)
            if not action:
                raise InvalidModel(self.domain_filename)

            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)

            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)
            # not applicable
            if(len(invalid)>0):
                executability = i/len(action_sequence)
                # for prec in invalid:
                #     print(prec.predicate, end= "|")
                # print("", end=",")

                if debug:
                    print(f"action {op.name} not executable")
                    print("preconditions not satisfied: ", invalid)
                    print("ending with executability: ", executability)
                    
                return executability
            # apply action
            adds = set(e for _,e in op.add_effects)
            dels = set(e for _,e in op.del_effects)
            # mark visited effects
            visited = visited.union(adds).union(dels);

            true_effs = true_effs.union(adds)
            true_effs.difference_update(dels)

          
            if debug:
                print(f"{op.name}... executed")
                print("adding:")
                print([e for _,e in op.add_effects])
                print("deleting:")
                print([e for _,e in op.del_effects])
                print()
        return 1
            




