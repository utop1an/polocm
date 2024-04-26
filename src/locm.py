import file_reader
import os
from collections import defaultdict
from utils import *


class Transition:
    raw_transistions = set()
    raw_arguments = set()
    raw_actions = set()
    types = set()
    raw_actarg = defaultdict(list)
    types = set()
    argtype = defaultdict(set)

    def __init__(self, seqs) -> None:
        # Todo: resolve missing/noise 
        self.parse_seqs(seqs)


    def parse_seqs(self,seqs):
        for seq in seqs:
            for actarg_tuple in seq:
                self.raw_actions.add(actarg_tuple[0])
                self.raw_actarg[actarg_tuple[0]].append(actarg_tuple[1])
                for j, arg in enumerate(actarg_tuple[1]):
                    self.raw_transistions.add(actarg_tuple[0]+"."+str(j))
                    self.raw_arguments.add(arg)

    def get_types(self):
        pass

    def build_transition_graphs(self):
        pass

# test
seqs = file_reader.read_action_seqs('test_input.txt')
t = Transition(seqs)
print(t.raw_actions)
print()
print(t.raw_arguments)
print()
print(t.raw_transistions)
print()
print(t.raw_actarg.items())