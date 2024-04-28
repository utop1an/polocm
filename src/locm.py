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
    raw_types = []
    argtype = defaultdict(set)

    def __init__(self, seqs) -> None:
        # Todo: resolve missing/noise 
        self.parse_seqs(seqs)
        self.get_types()


    def parse_seqs(self,seqs):
        for seq in seqs:
            for actarg_tuple in seq:
                self.raw_actions.add(actarg_tuple[0])
                self.raw_actarg[actarg_tuple[0]].append(actarg_tuple[1])
                for j, arg in enumerate(actarg_tuple[1]):
                    self.raw_transistions.add(actarg_tuple[0]+"."+str(j))
                    self.raw_arguments.add(arg)

    # class util functions.
    def get_types(self):
        # TODO incorporate word similarity in get classes.
        c = defaultdict(set)
        for k,v in self.raw_actarg.items():
            for arg_list in v:
                for i,object in enumerate(arg_list):
                    c[k,i].add(object)

        sets = c.values()
        print('sets:\n')
        print(sets)
        types = []
        # remove duplicate classes
        for s in sets:
            if s not in types:
                types.append(s)
        print('types:', types)
        # now do pairwise intersections of all values. If intersection, combine them; then return the final sets.
        types_cp = list(types)
        while True:
            combinations = list(itertools.combinations(types_cp,2))
            print('combin\n', combinations)
            intersections_count = 0
            for combination in combinations:
                if combination[0].intersection(combination[1]):
                    intersections_count +=1

                    if combination[0] in types_cp:
                        types_cp.remove(combination[0])
                    if combination[1] in types_cp:
                        types_cp.remove(combination[1])
                    types_cp.append(combination[0].union(combination[1]))

            if intersections_count==0:
                # print("no intersections left")
                break

        self.raw_types = types_cp

    # TODO: Can use better approach here. NER might help.
    def get_class_names(classes):
        # Name the class to first object found ignoring the digits in it
        class_names = []
        for c in classes:
            for object in c:
    #             object = ''.join([i for i in object if not i.isdigit()])
                class_names.append(object)
                break
        return class_names

    def get_class_index(arg,classes):
        for class_index, c in enumerate(classes):
            if arg in c:
                return class_index #it is like breaking out of the loop
        print("Error:class index not found") #this statement is only executed if class index is not returned.

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
print()
print(t.raw_types)