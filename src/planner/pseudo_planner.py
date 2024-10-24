
from math import e
import pddl_parser
import translate.normalize as normalize
import os
from utils.common_errors import InvalidModel, InvalidActionSequence
from utils import (
    set_timer_throw_exc,
    GeneralTimeOut
)

class PseudoPlanner:
    def __init__(self, domain_filename, executability_type='overall') -> None:
        self.domain_filename = domain_filename
        try:
            self.initialize_task()
            self.executability_type = executability_type
        except Exception as e:
            raise InvalidModel(domain_filename, e)

    @set_timer_throw_exc(num_seconds=30, exception=GeneralTimeOut, max_time=30, source="pseudo planner")
    def initialize_task(self):
        """Initialize the task from domain and problem files. This should be called in the worker process."""
        domain = pddl_parser.open(self.domain_filename)
        normalize.normalize(domain)
        self.domain = domain

    def check_executability(self,action_sequence, debug=False):
        if (self.executability_type == 'overall'):
            return self.get_overall_executability(action_sequence, debug)
        elif (self.executability_type == 'first_fail'):
            return self.get_first_fail_executability(action_sequence, debug)
        else:
            return self.get_overall_executability(action_sequence, debug)
    
    def get_overall_executability(self,action_sequence, debug=False):
        if not self.domain:
            raise InvalidModel(self.domain_filename)
        if (len(action_sequence)==0):
            raise InvalidActionSequence("Length 0")
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
            if ('sort0' in param_types):
                index= param_types.index('sort0')
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
            raise InvalidModel(self.domain_filename)
        if (len(action_sequence)==0):
            raise InvalidActionSequence("Length 0")
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
            if ('sort0' in param_types):
                index= param_types.index('sort0')
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
            




