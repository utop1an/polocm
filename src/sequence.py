from typing import Any

# trivial?
class ArgType: # Arg Type
    name: str

class Argument: # Objects...
    raw_name: str
    name: str

    type: ArgType
    types = [] # probs of being different ArgTypes?

    def __init__(self, raw_arg) -> None:
        self.raw_name = raw_arg

    def __getattribute__(self, name: str) -> Any:
        if (name == 'name'):
            if (not self.name):
                print("Name of argument {0} not validated...".format(self.raw_name))
                return False

class ActType:
    name: str
    arg_types = []
    args_length = 0
    
class Action:
    raw_name: str
    name: str
    args = set()

    args_length = 0

    def __init__(self, act_tuple) -> None:
        self.raw_name = act_tuple[0]
        self.args_length = len(act_tuple[1])
        for arg in act_tuple[1]:
            arg = Argument(arg)
            self.args.add(arg)

    def __getattribute__(self, name: str) -> Any:
        if (name == 'name'):
            if (not self.name):
                print("Name of action {0} not validated...".format(self.raw_name))
                return False
    
class Sequence:
    raw_seq = []
    
    actions: list[Action] = []

    def __init__(self, raw_seq) -> None:
        self.raw_seq = raw_seq
        for act_tuple in raw_seq:
            act = Action(act_tuple)
            self.actions.append(act)
            self.args.union(act.args)

    def get_orders(self):
        pass

class FOSequence(Sequence):
    def __init__(self, raw_seq) -> None:
        super.__init__(self, raw_seq)

    def get_orders(self):
        return [self.actions]
        
        
# TODO implement for partial ordered action traces
class POSequence(Sequence):
    def __init__(self, raw_seq) -> None:
        super().__init__(raw_seq)
        # 

    def get_orders(self):
        pass
    

class Problem:
    raw_seqs = []

    seqs = []



    def __init__(self, raw_seqs, ordering) -> None:
        self.raw_seqs = raw_seqs
        for raw_seq in raw_seqs:
            if (ordering == "FO"):
                seq = FOSequence(raw_seq)
            elif (ordering == "PO"):
                seq = POSequence(raw_seq)
            self.seqs.append(seq)









