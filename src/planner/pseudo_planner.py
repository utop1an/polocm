
import pddl_parser

class PseudoPlanner:
    def __init__(self, domain_filename) -> None:
        self.domain =pddl_parser.open(domain_filename)
    
    def check_executability(self,action_sequence):
        add_effs = set()
        del_effs = set()
        for i, a in enumerate(action_sequence):
            action = self.domain.get_action(a.name)
            assert action, "Invalid action sequence with wrong action name!"
            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('sort0' in param_types):
                index= param_types['sort0']
                params.insert(index, '0')
            var_mapping = dict(zip(param_names, params))
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping, objects_by_type)

            # check applicable
            preconditions = set(op.precondition)
            if(len(preconditions.intersection(del_effs))>0):
                return (i+1)/len(a)
            # apply action
            adds = set([e for _,e in op.add_effects])
            dels = set([e for _,e in op.del_effects])

            add_effs = add_effs.union(adds)
            del_effs = del_effs.union(dels)

            add_effs.difference_update(dels)
            del_effs.difference_update(adds)
            
        return 1
            




