from collections import defaultdict
import itertools
import os
from tabulate import tabulate
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
pd.options.display.max_columns = 100
from IPython.display import display, Markdown
from ipycytoscape import *
from utils import *

class Object: # Objects...
    raw_name: str
    name: str

    sort: str
    sorts = [] # probs of being different Sorts?

    def __init__(self, raw_arg) -> None:
        self.raw_name = raw_arg


class Action:
    raw_name: str
    name: str
    args = []


    def __init__(self, act_tuple) -> None:
        self.raw_name = act_tuple[0]
        for arg in act_tuple[1]:
            arg = Object(arg)
            self.args.append(arg)

class ActionPosition:
    action: Action
    index: int
    sort: str

class ActionSequence:
    raw_seq = []
    
    actions: list[Action] = []
    act_names = set()
    obj_names = set()



    def __init__(self, raw_seq) -> None:
        self.raw_seq = raw_seq
        for act_tuple in raw_seq:
            act = Action(act_tuple)
            self.actions.append(act)
            self.args.union(act.args)

    def get_orders(self):
        pass

class FOActionSequence(ActionSequence):
    def __init__(self, raw_seq) -> None:
        super.__init__(self, raw_seq)

    def get_orders(self):
        return [self.actions]
        
        
# TODO implement for partial ordered action traces
class POActionSequence(ActionSequence):
    def __init__(self, raw_seq) -> None:
        super().__init__(raw_seq)
        # 

    def get_orders(self):
        pass
    
class State:
    predicate: str
    objects = []

class StateSequence():
    raw_seq = []
    states: list[State] = []

class LearningProblem:

    sorts = defaultdict(list)
    sort_names = []
    predicate_sorts = defaultdict(list)
    adjacency_matrix_list = []
    predicate_adjacency_matrix_list = []
    adjacency_matrix_list_with_holes = []
    predicate_adjacency_matrix_list_with_holes = []
    

    def __init__(self, raw_action_seqs,raw_state_seqs, domain_name="DOMAIN") -> None:
        self.domain_name = domain_name
        # read action seqs
        self.raw_action_seqs = raw_action_seqs
        self.action_seqs = []
        self.action_names = set()
        self.objects = set()
        self.transitions = set()
        self.actobj_dict = defaultdict(list)

        for raw_action_seq in raw_action_seqs.split("\n"):

            raw_action_seq = raw_action_seq.replace(" ", "").lower()
            if raw_action_seq and not raw_action_seq.isspace() and len(raw_action_seq)>1:
                action_seq = []
                action_defs = raw_action_seq.split("),")
                for action_def in action_defs:
                    action = action_def.split("(")[0].strip(")")
                    self.action_names.add(action)
                    
                    actargs = action_def.split("(")[1].strip(")")
                    actarg_list = actargs.split(",")
                    action_seq.append((action, actarg_list))
                    self.actobj_dict[action].append(actarg_list)
                    for i, arg in enumerate(actarg_list):
                        self.objects.add(arg)
                        self.transitions.add(action + "."+ str(i))
                
            
                self.action_seqs.append(action_seq)
        # read state seqs
        self.raw_state_seqs = raw_state_seqs
        self.state_seqs = []
        self.stobj_dict = defaultdict(list)
        self.predicate_names = set()
        state_seqs = []
        for raw_state_seq in raw_state_seqs.split("\n"):
            raw_state_seq = raw_state_seq.replace(" ", "").lower()
            if raw_state_seq and not raw_state_seq.isspace() and len(raw_state_seq)>1:
                state_seq = []
                states = raw_state_seq.split("],")
                for state in states:
                    predicate_list = []
                    state = state.strip("[").strip("]")
                    predicate_defs = state.split("),")
                    for predicate_def in predicate_defs:
                        predicate = predicate_def.split("(")[0].strip(")")
                        self.predicate_names.add(predicate)
                        pargs = predicate_def.split("(")[1].strip(")")
                        if pargs == "":
                            parg_list = []
                        else:
                            parg_list = pargs.split(",")
                        predicate_list.append((predicate, parg_list))
                        self.stobj_dict[predicate].append(parg_list)
                        
                    state_seq.append(predicate_list)
                state_seqs.append(state_seq)
        
        # complete state seqs?
        state_seqs = self.complete_state_seqs(state_seqs)
        self.state_seqs = state_seqs
        self.get_sorts()
       
    def complete_state_seqs(self, state_seqs):
        """
        If a predicate P(o_1,...,o_n) is true at some step t, it remains true in the state for any t' > t,
        till an action A(o_1,...,o_m) is executed, where  {o_1,...,o_n} is a subset of {o_1,...,o_m}
        """
        new_state_seqs = state_seqs.copy()
        for x, seq in enumerate(new_state_seqs):
            for y, state in enumerate(seq):
                for z in range(len(state)):
                    if (y<len(self.action_seqs[x])):
                        act, actarg_list = self.action_seqs[x][y]
                        p, parg_list = state[z]
                        if not set(parg_list).issubset(set(actarg_list)):
                            new_state_seqs[x][y+1].append(state[z])

        return new_state_seqs
    

    def get_sorts(self):
        # TODO incorporate word similarity in get classes.
        s = defaultdict(set)
        for k,v in self.actobj_dict.items():
            for obj_list in v:
                for i,object in enumerate(obj_list):
                    s[k,i].add(object)

        sets = s.values()
        sorts = []
        # remove duplicate classes
        for se in sets:
            if se not in sorts:
                sorts.append(se)

        # now do pairwise intersections of all values. If intersection, combine them; then return the final sets.
        sorts_copy = list(sorts)
        while True:
            combinations = list(itertools.combinations(sorts_copy,2))
            intersections_count = 0
            for combination in combinations:
                if combination[0].intersection(combination[1]):
                    intersections_count +=1

                    if combination[0] in sorts_copy:
                        sorts_copy.remove(combination[0])
                    if combination[1] in sorts_copy:
                        sorts_copy.remove(combination[1])
                    sorts_copy.append(combination[0].union(combination[1]))

            if intersections_count==0:
                # print("no intersections left")
                break
        
        # add zero class
        sorts_copy.insert(0,{'zero'})
        self.sorts = sorts_copy

        # get sort names
        sort_names = []
        for sort in self.sorts:
            # TODO: use LLM to generate better sort name
            for object in sort:
                sort_names.append(object)
                break
        self.sort_names= sort_names
        
        # get predicate_sort
        predicate_sorts = dict()
        for seq in self.state_seqs:
            for state in seq:
                for predicate in state:
                    if predicate[0] not in predicate_sorts.keys():
                        predicate_sorts[predicate[0]] = [self.get_sort_index(x) for x in predicate[1]]
        self.predicate_sorts = predicate_sorts
      

    def get_sort_index(self,obj):
        for i, s in enumerate(self.sorts):
            if obj in s:
                return i
        print("Error:class index not found") 
        return False
    
    def get_predicate_sort_index(self, predicate_name):
        for i, p in enumerate(self.predicate_sorts.keys()):
            if p == predicate_name:
                return i
        return False
    
    def build_transition_graphs(self):
        # There should be a graph for each class of objects.
        graphs = []
        # Initialize all graphs empty
        for sort in self.sorts:
            graphs.append(nx.DiGraph())

        predicate_graphs = []
        for _ in self.predicate_sorts.keys():
            predicate_graphs.append(nx.DiGraph())

        consecutive_transition_lists = [] #list of consecutive transitions per object instance per sequence.
        
        # build transitions for single sort
        for m, obj in enumerate(self.objects):  # for all arguments (objects found in sequences)
            for n, seq in enumerate(self.action_seqs):  # for all sequences
                consecutive_transition_list = list()  # consecutive transition list for a sequence and an object (arg)
                for i, actobj_tuple in enumerate(seq):
                    for j, obj_prime in enumerate(actobj_tuple[1]):  # for all arguments in actarg tuples
                        if obj == obj_prime:  # if argument matches arg
                            node = actobj_tuple[0] + "." +  str(j)
                            # node = actarg_tuple[0] +  "." + class_names[get_class_index(arg,classes)] + "." +  str(j)  # name the node of graph which represents a transition
                            consecutive_transition_list.append(node)  # add node to the cons_transition for sequence and argument

                            # for each class append the nodes to the graph of that class
                            sort_index = self.get_sort_index(obj_prime)  # get index of class to which the object belongs to
                            graphs[sort_index].add_node(node)  # add node to the graph of that class

                consecutive_transition_lists.append([n, obj, consecutive_transition_list])
        
        # build transitions for Zero obj
        for n, seq in enumerate(self.action_seqs):  # for all sequences
            consecutive_transition_list = list()  # consecutive transition list for a sequence and an object (arg)
            for i, actobj_tuple in enumerate(seq):
                node = actobj_tuple[0]+'.-1'
                consecutive_transition_list.append(node)
                # add node to the graph of that class

            consecutive_transition_lists.append([n, 'zero', consecutive_transition_list])

        # build transitions for group of sorts evidenced as args in predicates
        consecutive_predicate_transition_lists = []
        for x, seq in enumerate(self.state_seqs): # for all state seqs
            for y, state in enumerate(seq): # for all states in each state seq
                for z, predicate in enumerate(state): # for all predicates in each state
                    for n, seq in enumerate(self.action_seqs):  # for all action sequences

                        consecutive_predicate_transition_list = []
                        for i, actobj_tuple in enumerate(seq): # for all action in each action seq
                            
                            if (set(predicate[1]).issubset(set(actobj_tuple[1]))):  # fix: if parg is subset of actarg --- if predicate arg matches action arg --- 
                                
                                node = actobj_tuple[0] + "." +  predicate[0]
                                print("adding transition node {0}, args: {1} and {2}".format(node, predicate[1], [actobj_tuple[1]]))
                                # node = actarg_tuple[0] +  "." + class_names[get_class_index(arg,classes)] + "." +  str(j)  # name the node of graph which represents a transition
                                consecutive_predicate_transition_list.append(node)  # add node to the cons_transition for sequence and argument

                                # for each class append the nodes to the graph of that class
                                predicate_sort_index = self.get_predicate_sort_index(predicate[0])  # get index of class to which the object belongs to
                                predicate_graphs[predicate_sort_index].add_node(node)  # add node to the graph of that class

                        consecutive_predicate_transition_lists.append([n, predicate[0], consecutive_predicate_transition_list])

        # for all consecutive transitions add edges to the appropriate graphs.
        for cons_trans_list in consecutive_transition_lists:
            # print(cons_trans_list)
            seq_no = cons_trans_list[0]  # get sequence number
            obj = cons_trans_list[1]  # get argument
            sort_index = self.get_sort_index(obj)  # get index of class
            # add directed edges to graph of that class
            for i in range(0, len(cons_trans_list[2]) - 1):
                    if graphs[sort_index].has_edge(cons_trans_list[2][i], cons_trans_list[2][i + 1]):
                        graphs[sort_index][cons_trans_list[2][i]][cons_trans_list[2][i + 1]]['weight'] += 1
                    else:
                        graphs[sort_index].add_edge(cons_trans_list[2][i], cons_trans_list[2][i + 1], weight=1)

        for cptl in consecutive_predicate_transition_lists:
            predicate_name = cptl[1]
            predicate_sort_index = self.get_predicate_sort_index(predicate_name)
            for i in range(0, len(cptl[2])-1):
                if predicate_graphs[predicate_sort_index].has_edge(cptl[2][i], cptl[2][i + 1]):
                    predicate_graphs[predicate_sort_index][cptl[2][i]][cptl[2][i + 1]]['weight'] += 1
                else:
                    predicate_graphs[predicate_sort_index].add_edge(cptl[2][i], cptl[2][i + 1], weight=1)

        
        # save all the graphs
        adjacency_matrix_list = [] # list of adjacency matrices per class
        for index, G in enumerate(graphs):
            df = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            adjacency_matrix_list.append(df)
        self.adjacency_matrix_list = adjacency_matrix_list # list of adjacency matrices per class
       
        # plot cytoscape interactive graphs
        cytoscapeobs = plot_cytographs(graphs, self.sort_names, self.adjacency_matrix_list)
        
        predicate_adjacency_matrix_list = []
        for _,G in enumerate(predicate_graphs):
            df = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            predicate_adjacency_matrix_list.append(df)
        self.predicate_adjacency_matrix_list = predicate_adjacency_matrix_list
        predicate_cytoscapeobs = plot_cytographs(predicate_graphs, list(self.predicate_sorts.keys()), self.predicate_adjacency_matrix_list)

    def locm2(self):
        self.adjacency_matrix_list_with_holes = self.get_adjacency_matrix_with_holes(self.adjacency_matrix_list)
        # self.predicate_adjacency_matrix_list_with_holes = self.get_adjacency_matrix_with_holes(self.predicate_adjacency_matrix_list)
        self.print_holes()

        self.holes_per_sort = self.get_holes_per_sort(self.adjacency_matrix_list_with_holes, self.sort_names)
        self.transitions_per_sort = self.get_transition_per_sort(self.adjacency_matrix_list_with_holes)
        self.consecutive_transitions_per_sort = self.get_consecutive_transitions_per_sort(self.adjacency_matrix_list_with_holes)

        self.transition_sets_per_sort = self.locm2_get_transition_sets_per_sort()

        


    def get_adjacency_matrix_with_holes(self, adjacency_matrix_list):
        adjacency_matrix_list_with_holes = []
        for index,adjacency_matrix in enumerate(adjacency_matrix_list):
            df = adjacency_matrix.copy()
            df1 = adjacency_matrix.copy()

            # for particular adjacency matrix's copy, loop over all pairs of rows
            for i in range(df.shape[0] - 1):
                for j in range(i+1, df.shape[0]):
                    idx1, idx2 = i, j
                    row1, row2 = df.iloc[idx1,:], df.iloc[idx2, :] #we have now all pairs of rows

                    common_values_flag = False #for each two rows we have a common_values_flag

                    # if there is a common value between two rows, turn common value flag to true
                    for col in range(row1.shape[0]):
                        if row1.iloc[col] > 0 and row2.iloc[col] > 0:
                            common_values_flag = True
                            break

                    # now if two rows have common values, we need to check for holes.
                    if common_values_flag:
                        for col in range(row1.shape[0]):
                            if row1.iloc[col] > 0 and row2.iloc[col] == 0:
                                df1.iloc[idx2,col] = 'hole'
                            elif row1.iloc[col] == 0 and row2.iloc[col] > 0:
                                df1.iloc[idx1, col] = 'hole'

            adjacency_matrix_list_with_holes.append(df1)
        return adjacency_matrix_list_with_holes
    
    def get_holes_per_sort(self, aml_with_holes, sort_names):
        holes_per_sort = []
        for index,df in enumerate(aml_with_holes):
            holes = set()
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    if df.iloc[i,j] == 'hole':
                        holes.add(frozenset({df.index[i] , df.columns[j]}))
            holes_per_sort.append(holes)

        for i, hole in enumerate(holes_per_sort):
            print("#holes in sort " + sort_names[i]+":" + str(len(hole)))

        return holes_per_sort
        
        
        

    def get_transition_per_sort(self, aml_with_holes):
        transitions_per_sort = []
        for index, df in enumerate(aml_with_holes):
            transitions_per_sort.append(df.columns.values)
        return transitions_per_sort
    
    def get_consecutive_transitions_per_sort(self, aml_with_holes) :
        consecutive_transitions_per_sort = []
        for index, df in enumerate(aml_with_holes):
            consecutive_transitions = set()  # for a class
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    if df.iloc[i, j] != 'hole':
                        if df.iloc[i, j] > 0:
                            consecutive_transitions.add((df.index[i], df.columns[j]))
            consecutive_transitions_per_sort.append(consecutive_transitions)
        return consecutive_transitions_per_sort
    
    def check_well_formed(self,subset_df):
        # got the adjacency matrix subset
        df = subset_df.copy()
        well_formed_flag = True
        
        
        if (df == 0).all(axis=None): # all elements are zero
            well_formed_flag = False
            
        # for particular adjacency matrix's copy, loop over all pairs of rows
        for i in range(0, df.shape[0]-1):
            for j in range(i + 1, df.shape[0]):
                print(i,j)
                idx1, idx2 = i, j
                row1, row2 = df.iloc[idx1, :], df.iloc[idx2, :]  # we have now all pairs of rows

                common_values_flag = False  # for each two rows we have a common_values_flag

                # if there is a common value between two rows, turn common value flag to true
                for col in range(row1.shape[0]):
                    if row1.iloc[col] > 0 and row2.iloc[col] > 0:
                        common_values_flag = True
                        break
            
                if common_values_flag:
                    for col in range(row1.shape[0]): # check for holes if common value
                        if row1.iloc[col] > 0 and row2.iloc[col] == 0:
                            well_formed_flag = False
                        elif row1.iloc[col] == 0 and row2.iloc[col] > 0:
                            well_formed_flag = False
        
        if not well_formed_flag:
            return False
        elif well_formed_flag:
            return True
    
    def check_valid(self, subset_df,consecutive_transitions_per_class):
    
        # Note: Essentially we check validity against P instead of E. 
        # In the paper of LOCM2, it isn't mentioned how to check against E.
        
        # Reasoning: If we check against all consecutive transitions per class, 
        # we essentially check against all example sequences.
        # check the candidate set which is well-formed (subset df against all consecutive transitions)

        # got the adjacency matrix subset
        df = subset_df.copy()

        # for particular adjacency matrix's copy, loop over all pairs of rows
        for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                if df.iloc[i,j] > 0:
                    valid_val_flag = False
                    ordered_pair = (df.index[i], df.columns[j])
                    for ct_list in consecutive_transitions_per_class:
                        for ct in ct_list:
                            if ordered_pair == ct:
                                valid_val_flag=True
                    # if after all iteration ordered pair is not found, mark the subset as invalid.
                    if not valid_val_flag:
                        return False
                    
        # return True if all ordered pairs found.
        return True
     
    def locm2_get_transition_sets_per_sort(self):
        """LOCM 2 Algorithm in the original LOCM2 paper"""
        
        # contains Solution Set S for each class.
        transition_sets_per_sort = []

        # for each hole for a class/sort
        for index, holes in enumerate(self.holes_per_sort):
            sort_name = self.sort_names[index]
            printmd("### "+  sort_name)
            
            # S
            transition_set_list = [] #transition_sets_of_a_class, # intially it's empty
            
            if len(holes)==0:
                print("no holes") # S will contain just T_all
            
            if len(holes) > 0: # if there are any holes for a class
                print(str(len(holes)) + " holes")
                for ind, hole in enumerate(holes):
                    printmd("#### Hole " + str(ind + 1) + ": " + str(set(hole)))
                    is_hole_already_covered_flag = False
                    if len(transition_set_list)>0:
                        for s_prime in transition_set_list:
                            if hole.issubset(s_prime):
                                printmd("Hole "+ str(set(hole)) + " is already covered.")
                                is_hole_already_covered_flag = True
                                break
                        
                    # discover a set which includes hole and is well-formed and valid against test data.
                    # if hole is not covered, do BFS with sets of increasing sizes starting with s=hole
                    if not is_hole_already_covered_flag: 
                        h = hole.copy()
                        candidate_sets = []
                        # all subsets of T_all starting from hole's len +1 to T_all-1.
                        for i in range(len(h)+1,len(self.transitions_per_sort[index])): 
                            subsets = findsubsets(self.transitions_per_sort[index],i) # all subsets of length i

                            for s in subsets:
                                if h.issubset(s): # if  is subset of s
                                    candidate_sets.append(set(s))
                            
                            s_well_formed_and_valid = False
                            for s in candidate_sets:
                                if len(s)>=i:
                                    printmd("Checking candidate set *" + str(s) + "* of class **" + sort_name + "** for well formedness and Validity")
                                    subset_df = self.adjacency_matrix_list[index].loc[list(s),list(s)]
                                    print_table(subset_df)

                                    # checking for well-formedness
                                    well_formed_flag = False
                                    well_formed_flag = self.check_well_formed(subset_df)
                                    if not well_formed_flag:
                                        print("This subset is NOT well-formed")
                                        
                                    elif well_formed_flag:
                                        print("This subset is well-formed.")
                                        # if well-formed validate across the data E
                                        # to remove inappropriate dead-ends
                                        valid_against_data_flag = False
                                        valid_against_data_flag = self.check_valid(subset_df, self.consecutive_transitions_per_sort)
                                        if not valid_against_data_flag:
                                            print("This subset is well-formed but invalid against example data")

                                        if valid_against_data_flag:
                                            print("This subset is valid.")
                                            print("Adding this subset " + str(s) +" to the locm2 transition set.")
                                            if s not in transition_set_list: # do not allow copies.
                                                transition_set_list.append(s)
                                            
                                            print("Hole that is covered now:")
                                            print(list(h))
                                            s_well_formed_and_valid = True
                                            break 
                            if s_well_formed_and_valid:
                                    break
                                            
                                            

            print(transition_set_list)                                    
            #step 7 : remove redundant sets S - {s1}
            ts_copy = transition_set_list.copy()
            for i in range(len(ts_copy)):
                for j in range(len(ts_copy)):
                    if ts_copy[i] < ts_copy[j]: #if subset
                        if ts_copy[i] in transition_set_list:
                            transition_set_list.remove(ts_copy[i])
                    elif ts_copy[i] > ts_copy[j]:
                        if ts_copy[j] in transition_set_list:
                            transition_set_list.remove(ts_copy[j])
            print("\nRemoved redundancy transition set list")
            print(transition_set_list)

            #step-8: include all-transitions machine, even if it is not well-formed.
            transition_set_list.append(set(self.transitions_per_sort[index])) #fallback
            printmd("#### Final transition set list")
            print(transition_set_list)
            transition_sets_per_sort.append(transition_set_list)
        return transition_sets_per_sort
    
    def locm(self):
        self.state_machines_overall_list = self.unify_start_and_end()
        state_mappings_sort, state_machines_overall_list_2 = self.rename_state()
        self.state_mappings_sort = state_mappings_sort
        self.state_machines_overall_list_2 = state_machines_overall_list_2
        HS_list, ct_list = self.form_HS()
        self.HS_list = HS_list
        self.HS_list_retained = self.test_HS()
        self.param_bindings_list_overall = self.create_and_merge_state_params()
        self.param_bindings_overall_flaw_removed = self.remove_param_flaw()
    


    def unify_start_and_end(self):
        state_machines_overall_list = []

        for index, ts_sort in enumerate(self.transition_sets_per_sort):
            fsms_per_sort = []
            printmd("### "+ self.sort_names[index])
            num_fsms = len(ts_sort)
            print("Number of FSMS:" + str(num_fsms))
            
            for fsm_no, ts in enumerate(ts_sort):
                fsm_graph = nx.DiGraph()
                
                printmd("#### FSM " + str(fsm_no))
                for t in ts:
                    source = "s(" + str(t) + ")"
                    target = "e(" + str(t) + ")"
                    fsm_graph.add_edge(source,target,weight=t)
                
            
                t_df = self.adjacency_matrix_list[index].loc[list(ts), list(ts)] #transition df for this fsm
                print_table(t_df)
                
                
                # merge end(t1) = start(t2) from transition df
                
                edge_t_list = [] # edge transition list
                for i in range(t_df.shape[0]):
                    for j in range(t_df.shape[1]):
                        
                        if t_df.iloc[i, j] != 'hole':
                            if t_df.iloc[i, j] > 0:
                                for node in fsm_graph.nodes():
                                    if "e("+t_df.index[i]+")" in node:
                                        merge_node1 = node
                                    if "s("+t_df.index[j]+")" in node:
                                        merge_node2 = node
                                
                                
                                

                                fsm_graph = nx.contracted_nodes(fsm_graph, merge_node1, merge_node2 , self_loops=True)

                                if merge_node1 != merge_node2:
                                    mapping = {merge_node1: merge_node1 + "|" + merge_node2} 
                                    fsm_graph = nx.relabel_nodes(fsm_graph, mapping)

                # we need to complete the list of transitions 
                # that can happen on self-loop nodes 
                # as these have been overwritten (as graph is not MultiDiGraph)
                
                sl_state_list = list(nx.nodes_with_selfloops(fsm_graph)) # self looping states.
                # if state is self-looping
                t_list = []
                if len(sl_state_list)>0: 
                    # if s(T1) and e(T1) are there for same node, this T1 can self-loop occur.
                    for s in sl_state_list:
                        for sub_s in s.split('|'):
                            if sub_s[0] == 'e':
                                if ('s' + sub_s[1:]) in s.split('|'):
                                    t_list.append(sub_s[2:-1])
                        fsm_graph[s][s]['weight'] = '|'.join(t_list)
                
                

                    
                plot_cytographs_fsm(fsm_graph,self.domain_name)
                df = nx.to_pandas_adjacency(fsm_graph, nodelist=fsm_graph.nodes(), weight = 1)
                print_table(df)
                fsms_per_sort.append(fsm_graph)
            state_machines_overall_list.append(fsms_per_sort)
        return state_machines_overall_list

    def rename_state(self):
        # An Automatic state dictionary is added here where states are 
        # renamed as 0, 1, 2 etc. for a specific FSM

        state_mappings_sort = []
        state_machines_overall_list_2 = []
        for index, fsm_graphs in enumerate(self.state_machines_overall_list):
            state_mappings_fsm = []
            fsms_per_sort_2 = []
            printmd("### "+ self.sort_names[index])
            num_fsms = len(fsm_graphs)
            print("Number of FSMS:" + str(num_fsms))
            
            for fsm_no, G in enumerate(fsm_graphs):
                
                state_mapping = {k: v for v, k in enumerate(G.nodes())}
                G_copy = nx.relabel_nodes(G, state_mapping)
                
                plot_cytographs_fsm(G, self.domain_name)
                plot_cytographs_fsm(G_copy, self.domain_name)
                printmd("Fsm "+ str(fsm_no))
                fsms_per_sort_2.append(G_copy)
                state_mappings_fsm.append(state_mapping)
                
            state_machines_overall_list_2.append(fsms_per_sort_2)
            state_mappings_sort.append(state_mappings_fsm)
        return state_mappings_sort, state_machines_overall_list_2
    
    def form_HS(self):
        HS_list = []
        ct_list = []

        # for transition set of each class
        for index, ts_class in enumerate(self.transition_sets_per_sort):
            printmd("### "+ self.sort_names[index])
            
            ct_per_sort = []
            HS_per_sort = []
            
            # for transition set of each fsm in a class
            for fsm_no, ts in enumerate(ts_class):
                printmd("#### FSM: " + str(fsm_no) + " Hypothesis Set")
                
                # transition matrix for the ts
                t_df = self.adjacency_matrix_list[index].loc[list(ts), list(ts)]
                ct_in_fsm = set()  # find consecutive transition set for a state machine in a class.
                for i in range(t_df.shape[0]):
                    for j in range(t_df.shape[1]):
                        if t_df.iloc[i, j] != 'hole':
                            if t_df.iloc[i, j] > 0:
                                ct_in_fsm.add((t_df.index[i], t_df.columns[j]))
                
                ct_per_sort.append(ct_in_fsm)
                
                # add to hypothesis set
                HS = set()
                
                # for each pair B.k and C.l in TS s.t. e(B.k) = S = s(C.l)
                for ct in ct_in_fsm:
                    B = ct[0].split('.')[0] # action name of T1
                    k = int(ct[0].split('.')[1]) # argument index of T1
                    
                    C = ct[1].split('.')[0] # action name of T2
                    l = int(ct[1].split('.')[1]) # argument index of T2
                    
                    
                    
                    
                    # When both actions B and C contain another argument of the same sort G' in position k' and l' respectively, 
                    # we hypothesise that there may be a relation between sorts G and G'.
                    for seq in self.action_seqs:
                        for actarg_tuple in seq:
                            arglist1 = []
                            arglist2 = []
                            if actarg_tuple[0] == B: #if action name is same as B
                                arglist1 = actarg_tuple[1].copy()
        #                         arglist1.remove(actarg_tuple[1][k]) # remove k from arglist
                                for actarg_tuple_prime in seq: #loop through seq again.
                                    if actarg_tuple_prime[0] == C:
                                        arglist2 = actarg_tuple_prime[1].copy()
        #                                 arglist2.remove(actarg_tuple_prime[1][l]) # remove l from arglist
                                        

                                # for arg lists of actions B and C, if class is same add a hypothesis set.
                                for i in range(len(arglist1)): # if len is 0, we don't go in
                                    for j in range(len(arglist2)):
                                        class1 = self.get_sort_index(arglist1[i])
                                        class2 = self.get_sort_index(arglist2[j])
                                        if class1 == class2: # if object at same position have same classes
                                            # add hypothesis to hypothesis set.
                                            if (k!=i) and (l!=j):
                                                HS.add((frozenset({"e("+B+"."+ str(k)+")", "s("+C+"."+str(l)+")"}),B,k,i,C,l,j,self.sort_names[index],self.sort_names[class1]))
                print(str(len(HS))+ " hypothesis created")
        #         for h in HS:
        #             print(h)
                
                HS_per_sort.append(HS)
            HS_list.append(HS_per_sort)
            ct_list.append(ct_per_sort)
        
        return HS_list, ct_list
    
    def test_HS(self):
        HS_list_retained = []
        for index, HS_sort in enumerate(self.HS_list):
            printmd("### "+ self.sort_names[index])
            HS_per_sort_retained = []


            for fsm_no, HS in enumerate(HS_sort):
                printmd("#### FSM: " + str(fsm_no) + " Hypothesis Set")

                count=0
                HS_copy = HS.copy()
                HS_copy2 = HS.copy()

                
                # for each object O occuring in Ou
                for O in self.objects:
                    #   for each pair of transitions Ap.m and Aq.n consecutive for O in seq
                    ct = []
                    for seq in self.action_seqs:
                        for actarg_tuple in seq:
                            act = actarg_tuple[0]
                            for j, arg in enumerate(actarg_tuple[1]):
                                if arg == O:
                                    ct.append((act + '.' + str(j), actarg_tuple[1]))


                    for i in range(len(ct)-1):
                        A_p = ct[i][0].split('.')[0]
                        m = int(ct[i][0].split('.')[1])
                        A_q = ct[i+1][0].split('.')[0]
                        n = int(ct[i+1][0].split('.')[1]) 

                        # for each hypothesis H s.t. A_p = B, m = k, A_q = C, n = l

                        for H in HS_copy2:
                            if A_p == H[1] and m == H[2] and A_q == H[4] and n == H[5]:
                                k_prime = H[3]
                                l_prime = H[6]

                                # if O_p,k_prime = Q_q,l_prime
                                if ct[i][1][k_prime] != ct[i+1][1][l_prime]:
                                    if H in HS_copy:
                                        HS_copy.remove(H)
                                        count += 1

                print(str(len(HS_copy))+ " hypothesis retained")
                # state machine
        #         if len(HS_copy)>0:
        #             plot_cytographs_fsm(state_machines_overall_list[index][fsm_no],domain_name)
        #         for H in HS_copy:
        #             print(H)
                HS_per_sort_retained.append(HS_copy)
            HS_list_retained.append(HS_per_sort_retained)
        return HS_list_retained
    
    def create_and_merge_state_params(self):
        # Each hypothesis refers to an incoming and outgoing transition 
        # through a particular state of an FSM
        # and matching associated transitions can be considered
        # to set and read parameters of a state.
        # Since there maybe multiple transitions through a give state,
        # it is possible for the same parameter to have multiple
        # pairwise occurences.

        print("Step 6: creating and merging state params")
        param_bindings_list_overall = []
        for sortindex, HS_per_sort in enumerate(self.HS_list_retained):
            param_bind_per_sort = []
            
            
            for fsm_no, HS_per_fsm in enumerate(HS_per_sort):
                param_binding_list = []
                
                # fsm in consideration
                G = self.state_machines_overall_list[sortindex][fsm_no]
                state_list = G.nodes()
                
                # creation
                for index,h in enumerate(HS_per_fsm):
                    param_binding_list.append((h,"v"+str(index)))
                
                merge_pl = [] # parameter to merge list
                if len(param_binding_list)>1:
                    # merging
                    pairs = findsubsets(param_binding_list, 2)
                    for pair in pairs:
                        h_1 = pair[0][0]
                        h_2 = pair[1][0]
                        
                        
                        # equate states
                        state_eq_flag = False
                        for s_index, state in enumerate(state_list):
                            # if both hyp states appear in single state in fsm
                            if list(h_1[0])[0] in state:
                                if list(h_1[0])[0] in state:
                                    state_eq_flag =True
                                    
                        
                        if ((state_eq_flag and h_1[1] == h_2[1] and h_1[2] == h_2[2] and h_1[3] == h_2[3]) or (state_eq_flag and h_1[4] == h_2[4] and h_1[5] == h_2[5] and h_1[6] == h_2[6])):
                            merge_pl.append(list([pair[0][1], pair[1][1]]))
                
                
            
                #inner lists to sets (to list of sets)
                l=[set(x) for x in merge_pl]

                #cartesian product merging elements if some element in common
                for a,b in itertools.product(l,l):
                    if a.intersection( b ):
                        a.update(b)
                        b.update(a)

                #back to list of lists
                l = sorted( [sorted(list(x)) for x in l])

                #remove dups
                merge_pl = list(l for l,_ in itertools.groupby(l))
                
                # sort
                for pos, l in enumerate(merge_pl):
                    merge_pl[pos] = sorted(l, key = lambda x: int(x[1:]))
                
                print(merge_pl) # equal params appear in a list in this list.
                
                    
                for z,pb in enumerate(param_binding_list):
                    for l in merge_pl:
                        if pb[1] in l:
                            # update pb
                            param_binding_list[z] = (param_binding_list[z][0], l[0])
                

                        
                
                param_bind_per_sort.append(param_binding_list)
                print(self.sort_names[sortindex])
                
                # set of params per class
                param = set()
                for pb in param_binding_list:
        #             print(pb)
                    param.add(pb[1])
                    
                # num of params per class
                printmd("No. of params earlier:" + str(len(param_binding_list)))
                printmd("No. of params after merging:" + str(len(param)))
                    
                
                
                
                
            param_bindings_list_overall.append(param_bind_per_sort)
        return param_bindings_list_overall
    
    def remove_param_flaw(self):
        # Removing State Params.
        # Flaw occurs Object can reach state S with param P having an inderminate value.
        # There is transition s.t. end(B.k) = S. 
        # but there is no h = <S,B,k,k',C,l,l',G,G') and <h,P> is in bindings.

        param_bindings_overall_flaw_removed  = []
        for sortindex, fsm_per_sort in enumerate(self.state_machines_overall_list):
            print(self.sort_names[sortindex])
            pb_per_sort_flaw_removed = []

            for fsm_no, G in enumerate(fsm_per_sort):
                
                pb_per_fsm_flaw_removed = []
                # G is fsm in consideration
                faulty_pb = []
                for state in G.nodes():
                    inedges = G.in_edges(state, data=True)
                    
                    for ie in inedges:
                        tr = ie[2]['weight']
                        t_list = tr.split('|')
                        for t in t_list:
                            B = t.split('.')[0]
                            k = t.split('.')[1]
                            S = 'e(' + t + ')'
                            flaw = True
                            for pb in self.param_bindings_list_overall[sortindex][fsm_no]:
                                H = pb[0]
                                v = pb[1]
                                if (S in set(H[0])) and (B==H[1]) and (int(k)==H[2]) :
                                    # this pb is okay
                                    flaw=False
        #                     print(flaw)
                            if flaw:
                                for pb in self.param_bindings_list_overall[sortindex][fsm_no]:
                                    H = pb[0]
                                    H_states = list(H[0])
                                    for h_state in H_states:
                                        if h_state in state:
                                            if pb not in faulty_pb:
                                                faulty_pb.append(pb) # no duplicates
                
                for pb in self.param_bindings_list_overall[sortindex][fsm_no]:
                    if pb not in faulty_pb:
                        pb_per_fsm_flaw_removed.append(pb)
                
                                        
                                
                                
                print(str(len(pb_per_fsm_flaw_removed)) + "/" + str(len(self.param_bindings_list_overall[sortindex][fsm_no])) + " param retained")
                for pb in pb_per_fsm_flaw_removed:
                    print(pb)

                        
                
                pb_per_sort_flaw_removed.append(pb_per_fsm_flaw_removed)
            param_bindings_overall_flaw_removed.append(pb_per_sort_flaw_removed)
        return param_bindings_overall_flaw_removed
    
    def toPDDL(self):
        # get action schema
        print(";;********************Learned PDDL domain******************")
        output_file = "output/" +  self.domain_name + ".pddl"
        write_file = open(output_file, 'w')
        write_line = "(define"
        write_line += "  (domain "+ self.domain_name+")\n"
        write_line += "  (:requirements :typing)\n"
        write_line += "  (:types"
        for sort_name in self.sort_names:
            write_line += " " + sort_name
        write_line += ")\n"
        write_line += "  (:predicates\n"

        # one predicate to represent each object state

        predicates = []
        for sort_index, pb_per_sort in enumerate(self.param_bindings_overall_flaw_removed):
            for fsm_no, pbs_per_fsm in enumerate(pb_per_sort):
                state_mapping = self.state_mappings_sort[sort_index][fsm_no]
                
                for state_index, state in enumerate(self.state_machines_overall_list[sort_index][fsm_no].nodes()):
                    
                    state_set = set(state.split('|'))
                    predicate = ""
            
                    write_line += "    (" + self.sort_names[sort_index] + "_fsm" + str(fsm_no) + "_state" +  str(state_mapping[state])
                    predicate += "    (" + self.sort_names[sort_index] + "_fsm" + str(fsm_no) + "_state" + str(state_mapping[state])
                    for pb in pbs_per_fsm:
                            if set(pb[0][0]) <= state_set:
                                if " ?"+pb[1] + " - " + str(pb[0][8]) not in predicate:
                                    write_line += " ?"+pb[1] + " - " + str(pb[0][8])
                                    predicate += " ?"+pb[1] + " - " + str(pb[0][8])
            
                    write_line += ")\n"
                    predicate += ")"
                    predicates.append(predicate)
        write_line += "  )\n"
                    
        for action_index, action in enumerate(self.action_names):
            write_line += "  (:action"
            write_line += "  " + action + " "
            write_line += "  :parameters"
            write_line += "  ("
            arg_already_written_flag = False
            params_per_action = []
            args_per_action = []
            for seq in self.action_seqs:
                for actarg_tuple in seq:
                    if not arg_already_written_flag:
                        if actarg_tuple[0] == action:
                            arglist = []
                            for arg in actarg_tuple[1]:
                                write_line += "?"+arg + " - " + self.sort_names[self.get_sort_index(arg)] + " "
                                arglist.append(arg)
                            args_per_action.append(arglist)
                            params_per_action.append(actarg_tuple[1])
                            arg_already_written_flag = True
            write_line += ")\n"


            # need to use FSMS to get preconditions and effects.
            # Start-state = precondition. End state= Effect
            preconditions = []
            effects = []
            for arglist in params_per_action:
                for arg in arglist:
                    current_sort_index = self.get_sort_index(arg)
                    for fsm_no, G in enumerate(self.state_machines_overall_list[current_sort_index]):
                        G_int = self.state_machines_overall_list_2[current_sort_index][fsm_no]
                        state_mapping = self.state_mappings_sort[current_sort_index][fsm_no]
                        for start, end, weight in G_int.edges(data='weight'):
                            _actions = weight.split('|')
                            for _action in _actions:
                                if _action.split('.')[0] == action:
                                    for predicate in predicates:
                                        pred = predicate.split()[0].lstrip("(")
                                        srt = pred.split('_')[0]
                                        fsm = pred.split('_')[1]
                                        state_ind = pred.split('_')[2].rstrip(")")[-1]

                                        if srt == self.sort_names[current_sort_index]:
                                            if fsm == "fsm" + str(fsm_no):
                                                if int(state_ind) == int(start):
                                                    if predicate not in preconditions:
                                                        preconditions.append(predicate)
                                                        
                                                if int(state_ind) == int(end):
                                                    if predicate not in effects:
                                                        effects.append(predicate)
                                    break
                                    

                        

            write_line += "   :precondition"
            write_line += "   (and\n"
            for precondition in preconditions:
                write_line += "    "+precondition+"\n"
            write_line += "   )\n"
            write_line += "   :effect"
            write_line += "   (and\n"
            for effect in effects:
                write_line += "    " + effect + "\n"
            write_line += "  )"

            write_line += ")\n\n"

        write_line += ")\n" #domain ending bracket


        print(write_line)

        write_file.write(write_line)
        write_file.close()



    def print_holes(self):
        print(len(self.adjacency_matrix_list))
        print(len(self.adjacency_matrix_list_with_holes))
        for index,adjacency_matrix in enumerate(self.adjacency_matrix_list):
            printmd("\n#### " + self.sort_names[index] )
            print_table(adjacency_matrix)

            printmd("\n#### HOLES: " + self.sort_names[index])
            print_table(self.adjacency_matrix_list_with_holes[index])

        # predicate_sorts_names = list(self.predicate_sorts.keys())
        # for index, pam in enumerate(self.predicate_adjacency_matrix_list):
        #     printmd("\n#### " + predicate_sorts_names[index] )
        #     print_table(pam)

        #     printmd("\n#### HOLES: " + predicate_sorts_names[index])
        #     print_table(self.predicate_adjacency_matrix_list_with_holes[index])

    def print_action_sequences(self):
        for seq in self.action_seqs:
            for index,action in enumerate(seq):
                print(str(index) + ": " + str(action))
            print()
    
    def print_state_sequences(self):
        for seq in self.state_seqs:
            for index,state in enumerate(seq):
                print(str(index) + ": " + str(state))
            print()
    
    def print_info(self):
        print("action names:\n",self.action_names)
        print()
        print("objects:\n",self.objects)
        print()
        print("sorts of obj:\n", self.sorts)
        print()
        print("transitions:\n", self.transitions)
        print()
        print("predicate names:\n", self.predicate_names)











