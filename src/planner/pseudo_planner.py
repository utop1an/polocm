
import pddl_parser
import pddl.normalize as normalize

class PseudoPlanner:
    def __init__(self, domain_filename) -> None:
        self.domain =pddl_parser.open(domain_filename)
        normalize.normalize(self.domain)
        
    def check_executability(self,action_sequence, debug=False):
        add_effs = set()
        del_effs = set()
        visited = set()
        for i, a in enumerate(action_sequence):
            action = self.domain.get_action(a.name)
            assert action, f"Invalid action sequence with wrong action name: {a.name}!"

            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('sort0' in param_types):
                index= param_types.index('sort0')
                params.insert(index, '0')
            var_mapping = dict(zip(param_names, params))
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping, objects_by_type)

            # check applicable
            preconditions = set(op.precondition)
            invalid_add = preconditions.difference(add_effs)
            invalid_add = invalid_add.intersection(visited)
            invalid_del = preconditions.intersection(del_effs)
            invalid = invalid_add.union(invalid_del)
            if(len(invalid)>0):
                executability = i/len(action_sequence)
                if debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)
                    print("ending with executability: ", executability)
                return executability
            # apply action
            adds = set([e for _,e in op.add_effects])
            dels = set([e for _,e in op.del_effects])
            
            visited = visited.union(adds).union(dels);

            add_effs = add_effs.union(adds)
            del_effs = del_effs.union(dels)

            add_effs.difference_update(dels)
            del_effs.difference_update(adds)
            if debug:
                print(f"action {op} executed")
        return 1
            




