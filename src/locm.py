import file_reader
import os
from collections import defaultdict
from utils import *


class Transition:
    domain_name= "DOMAIN"
    raw_seqs = []
    raw_transistions = set()
    raw_arguments = set()
    raw_actions = set()
    types = set()
    raw_actarg = defaultdict(list)
    types = set()
    raw_types = []
    argtype = defaultdict(set)

    def __init__(self, seqs, domain_name="DOMAIN_XXX") -> None:
        # Todo: resolve missing/noise 
        self.domain_name = domain_name
        self.raw_seqs = seqs
        self.parse_seqs(seqs)
        self.get_types()
        self.get_type_names()

    def locm(self):
        printmd("## "+ self.domain_name.upper())
        adjacency_matrix_list, graphs, cytoscapeobjs = self.build_and_save_transition_graphs()
        adjacency_matrix_list_with_holes = self.get_adjacency_matrix_with_holes(adjacency_matrix_list)
        self.dump_adjacency_matrix(adjacency_matrix_list, adjacency_matrix_list_with_holes)
        holes_per_class = self.get_holes_per_class(adjacency_matrix_list_with_holes)
        transitions_per_class = self.get_transitions_per_class(adjacency_matrix_list_with_holes)
        consecutive_transitions_per_class =  self.get_consecutive_transitions_per_class(adjacency_matrix_list_with_holes)
        printmd("### Getting transitions sets for each class using LOCM2")
        transition_sets_per_class = self.locm2_get_transition_sets_per_class(holes_per_class, transitions_per_class, consecutive_transitions_per_class, adjacency_matrix_list)
        state_machines_overall_list = self.mark_start_end_state(transition_sets_per_class, adjacency_matrix_list)
        state_mappings_class, state_machines_overall_list_2 = self.rename_states(state_machines_overall_list)
        HS_list = self.form_hs(transition_sets_per_class, adjacency_matrix_list)
        HS_list_retained = self.test_hs(HS_list)
        
        param_bindings_list_overall = self.create_and_merge_state_params(HS_list_retained, state_machines_overall_list)
        para_bind_overall_fault_removed = self.remove_param_flaws(param_bindings_list_overall, state_machines_overall_list)
        self.form_pddl(para_bind_overall_fault_removed, state_machines_overall_list)
        self.validate_and_fix_pddl(para_bind_overall_fault_removed, state_mappings_class, state_machines_overall_list, state_machines_overall_list_2)

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

        self.types = types_cp

    # TODO: Can use better approach here. NER might help.
    def get_type_names(self):
        # Name the class to first object found ignoring the digits in it
        type_names = []
        for t in self.types:
            for object in t:
    #             object = ''.join([i for i in object if not i.isdigit()])
                type_names.append(object)
                break
        self.type_names = type_names

    def get_type_index(self, arg):
        for type_index, t in enumerate(self.types):
            if arg in t:
                return type_index #it is like breaking out of the loop
        print("Error:class index not found for", arg) #this statement is only executed if class index is not returned.

    def build_and_save_transition_graphs(self):
        # There should be a graph for each class of objects.
        graphs = []
        # Initialize all graphs empty
        for sort in self.types:
            graphs.append(nx.DiGraph())

        consecutive_transition_lists = [] #list of consecutive transitions per object instance per sequence.

        for m, arg in enumerate(self.raw_arguments):  # for all arguments (objects found in sequences)
            for n, seq in enumerate(self.raw_seqs):  # for all sequences
                consecutive_transition_list = list()  # consecutive transition list for a sequence and an object (arg)
                for i, actarg_tuple in enumerate(seq):
                    for j, arg_prime in enumerate(actarg_tuple[1]):  # for all arguments in actarg tuples
                        if arg == arg_prime:  # if argument matches arg
                            node = actarg_tuple[0] + "." +  str(j)
                            # node = actarg_tuple[0] +  "." + class_names[get_type_index(arg,classes)] + "." +  str(j)  # name the node of graph which represents a transition
                            consecutive_transition_list.append(node)  # add node to the cons_transition for sequence and argument

                            # for each class append the nodes to the graph of that class
                            class_index = self.get_type_index(arg_prime)  # get index of class to which the object belongs to
                            graphs[class_index].add_node(node)  # add node to the graph of that class

                consecutive_transition_lists.append([n, arg, consecutive_transition_list])

        # print(consecutive_transition_lists)
        # for all consecutive transitions add edges to the appropriate graphs.
        for cons_trans_list in consecutive_transition_lists:
            # print(cons_trans_list)
            seq_no = cons_trans_list[0]  # get sequence number
            arg = cons_trans_list[1]  # get argument
            class_index = self.get_type_index(arg)  # get index of class
            # add directed edges to graph of that class
            for i in range(0, len(cons_trans_list[2]) - 1):
                    if graphs[class_index].has_edge(cons_trans_list[2][i], cons_trans_list[2][i + 1]):
                        graphs[class_index][cons_trans_list[2][i]][cons_trans_list[2][i + 1]]['weight'] += 1
                    else:
                        graphs[class_index].add_edge(cons_trans_list[2][i], cons_trans_list[2][i + 1], weight=1)


        
        # make directory if doesn't exist
        dirName = "output/"+ self.domain_name
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")
        empty_directory(dirName)
        
        # save all the graphs

        adjacency_matrix_list = save(graphs, self.type_names, self.domain_name) # list of adjacency matrices per class
        
        # plot cytoscape interactive graphs
        cytoscapeobs = plot_cytographs(graphs,self.domain_name,self.type_names, adjacency_matrix_list)
        
        return adjacency_matrix_list, graphs, cytoscapeobs
    
    def get_adjacency_matrix_with_holes(self,adjacency_matrix_list):
        adjacency_matrix_list_with_holes = []
        for index,adjacency_matrix in enumerate(adjacency_matrix_list):
            # print("\n ROWS ===========")
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
    
    def dump_adjacency_matrix(self,adjacency_matrix_list, adjacency_matrix_list_with_holes):
        # Printing FSM matrices with and without holes
        for index,adjacency_matrix in enumerate(adjacency_matrix_list):
            printmd("\n#### " + self.type_names[index] )
            print_table(adjacency_matrix)

            printmd("\n#### HOLES: " + self.type_names[index])
            print_table(adjacency_matrix_list_with_holes[index])

    def get_holes_per_class(self, adjacency_matrix_list_with_holes):
        # Create list of set of holes per class (H)
        holes_per_class = []

        for index,df in enumerate(adjacency_matrix_list_with_holes):
            holes = set()
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    if df.iloc[i,j] == 'hole':
                        holes.add(frozenset({df.index[i] , df.columns[j]}))
            holes_per_class.append(holes)
        for i, hole in enumerate(holes_per_class):
            print("#holes in class " + self.type_names[i]+":" + str(len(hole)))
        #     for h in hole:
        #         print(list(h))
        return holes_per_class
    
    def get_transitions_per_class(self,adjacency_matrix_list_with_holes):
        transitions_per_class = []
        for index, df in enumerate(adjacency_matrix_list_with_holes):
            transitions_per_class.append(df.columns.values)
        return transitions_per_class

    def get_consecutive_transitions_per_class(self,adjacency_matrix_list_with_holes):
        consecutive_transitions_per_class = []
        for index, df in enumerate(adjacency_matrix_list_with_holes):
            consecutive_transitions = set()  # for a class
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    if df.iloc[i, j] != 'hole':
                        if df.iloc[i, j] > 0:
    #                         print("(" + df.index[i] + "," + df.columns[j] + ")")
                            consecutive_transitions.add((df.index[i], df.columns[j]))
            consecutive_transitions_per_class.append(consecutive_transitions)
        return consecutive_transitions_per_class

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
    
    def check_valid(self,subset_df,consecutive_transitions_per_class):
    
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
    
    def locm2_get_transition_sets_per_class(self,holes_per_class, transitions_per_class, consecutive_transitions_per_class, adjacency_matrix_list):
        """LOCM 2 Algorithm in the original LOCM2 paper"""
        
        # contains Solution Set S for each class.
        transition_sets_per_class = []

        # for each hole for a class/sort
        for index, holes in enumerate(holes_per_class):
            type_name = self.type_names[index]
            printmd("### "+  type_name)
            
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
                        for i in range(len(h)+1,len(transitions_per_class[index])): 
                            subsets = findsubsets(transitions_per_class[index],i) # all subsets of length i

                            for s in subsets:
                                if h.issubset(s): # if  is subset of s
                                    candidate_sets.append(set(s))
                            
                            s_well_formed_and_valid = False
                            for s in candidate_sets:
                                if len(s)>=i:
                                    printmd("Checking candidate set *" + str(s) + "* of class **" + type_name + "** for well formedness and Validity")
                                    subset_df = adjacency_matrix_list[index].loc[list(s),list(s)]
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
                                        valid_against_data_flag = self.check_valid(subset_df, consecutive_transitions_per_class)
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
            transition_set_list.append(set(transitions_per_class[index])) #fallback
            printmd("#### Final transition set list")
            print(transition_set_list)
            transition_sets_per_class.append(transition_set_list)
            

        return transition_sets_per_class
    
    def mark_start_end_state(self, transition_sets_per_class, adjacency_matrix_list):
        state_machines_overall_list = []

        for index, ts_class in enumerate(transition_sets_per_class):
            fsms_per_class = []
            printmd("### "+ self.type_names[index])
            num_fsms = len(ts_class)
            print("Number of FSMS:" + str(num_fsms))
            
            for fsm_no, ts in enumerate(ts_class):
                fsm_graph = nx.DiGraph()
                
                printmd("#### FSM " + str(fsm_no))
                for t in ts:
                    source = "s(" + str(t) + ")"
                    target = "e(" + str(t) + ")"
                    fsm_graph.add_edge(source,target,weight=t)
                
            
                t_df = adjacency_matrix_list[index].loc[list(ts), list(ts)] #transition df for this fsm
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
                fsms_per_class.append(fsm_graph)
            state_machines_overall_list.append(fsms_per_class)
        return state_machines_overall_list

    def rename_states(self, state_machines_overall_list):
        # An Automatic state dictionary is added here where states are 
        # renamed as 0, 1, 2 etc. for a specific FSM

        state_mappings_class = []
        state_machines_overall_list_2 = []
        for index, fsm_graphs in enumerate(state_machines_overall_list):
            state_mappings_fsm = []
            fsms_per_class_2 = []
            printmd("### "+ self.type_names[index])
            num_fsms = len(fsm_graphs)
            print("Number of FSMS:" + str(num_fsms))
            
            for fsm_no, G in enumerate(fsm_graphs):
                
                state_mapping = {k: v for v, k in enumerate(G.nodes())}
                G_copy = nx.relabel_nodes(G, state_mapping)
                
                plot_cytographs_fsm(G, self.domain_name)
                plot_cytographs_fsm(G_copy, self.domain_name)
                printmd("Fsm "+ str(fsm_no))
                fsms_per_class_2.append(G_copy)
                state_mappings_fsm.append(state_mapping)
                
            state_machines_overall_list_2.append(fsms_per_class_2)
            state_mappings_class.append(state_mappings_fsm)
        return state_mappings_class, state_machines_overall_list_2

    def form_hs(self, transition_sets_per_class, adjacency_matrix_list):
        HS_list = []
        ct_list = []

        # for transition set of each class
        for index, ts_class in enumerate(transition_sets_per_class):
            printmd("### "+ self.type_names[index])
            
            ct_per_class = []
            HS_per_class = []
            
            # for transition set of each fsm in a class
            for fsm_no, ts in enumerate(ts_class):
                printmd("#### FSM: " + str(fsm_no) + " Hypothesis Set")
                
                # transition matrix for the ts
                t_df = adjacency_matrix_list[index].loc[list(ts), list(ts)]
                ct_in_fsm = set()  # find consecutive transition set for a state machine in a class.
                for i in range(t_df.shape[0]):
                    for j in range(t_df.shape[1]):
                        if t_df.iloc[i, j] != 'hole':
                            if t_df.iloc[i, j] > 0:
                                ct_in_fsm.add((t_df.index[i], t_df.columns[j]))
                
                ct_per_class.append(ct_in_fsm)
                
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
                    for seq in self.raw_seqs:
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
                                        class1 = self.get_type_index(arglist1[i])
                                        class2 = self.get_type_index(arglist2[j])
                                        if class1 == class2: # if object at same position have same classes
                                            # add hypothesis to hypothesis set.
                                            if (k!=i) and (l!=j):
                                                HS.add((frozenset({"e("+B+"."+ str(k)+")", "s("+C+"."+str(l)+")"}),B,k,i,C,l,j,self.type_names[index],self.type_names[class1]))
                print(str(len(HS))+ " hypothesis created")
        #         for h in HS:
        #             print(h)
                
                HS_per_class.append(HS)
            HS_list.append(HS_per_class)
            ct_list.append(ct_per_class)
        return HS_list

    def test_hs(self,HS_list):
        HS_list_retained = []
        for index, HS_class in enumerate(HS_list):
            printmd("### "+ self.type_names[index])
            HS_per_class_retained = []


            for fsm_no, HS in enumerate(HS_class):
                printmd("#### FSM: " + str(fsm_no) + " Hypothesis Set")

                count=0
                HS_copy = HS.copy()
                HS_copy2 = HS.copy()

                
                # for each object O occuring in Ou
                for O in self.raw_arguments:
                    #   for each pair of transitions Ap.m and Aq.n consecutive for O in seq
                    ct = []
                    for seq in self.raw_seqs:
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
                HS_per_class_retained.append(HS_copy)
            HS_list_retained.append(HS_per_class_retained)
        return HS_list_retained
    
    def create_and_merge_state_params(self,HS_list_retained, state_machines_overall_list):
        param_bindings_list_overall = []
        for classindex, HS_per_class in enumerate(HS_list_retained):
            param_bind_per_class = []
            
            
            for fsm_no, HS_per_fsm in enumerate(HS_per_class):
                param_binding_list = []
                
                # fsm in consideration
                G = state_machines_overall_list[classindex][fsm_no]
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
                

                        
                
                param_bind_per_class.append(param_binding_list)
                print(self.type_names[classindex])
                
                # set of params per class
                param = set()
                for pb in param_binding_list:
        #             print(pb)
                    param.add(pb[1])
                    
                # num of params per class
                printmd("No. of params earlier:" + str(len(param_binding_list)))
                printmd("No. of params after merging:" + str(len(param)))
            param_bindings_list_overall.append(param_bind_per_class)
        return param_bindings_list_overall
    
    def remove_param_flaws(self, param_bindings_list_overall, state_machines_overall_list):
        para_bind_overall_fault_removed  = []
        for classindex, fsm_per_class in enumerate(state_machines_overall_list):
            print(self.type_names[classindex])
            pb_per_class_fault_removed = []

            for fsm_no, G in enumerate(fsm_per_class):
                
                pb_per_fsm_fault_removed = []
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
                            for pb in param_bindings_list_overall[classindex][fsm_no]:
                                H = pb[0]
                                v = pb[1]
                                if (S in set(H[0])) and (B==H[1]) and (int(k)==H[2]) :
                                    # this pb is okay
                                    flaw=False
        #                     print(flaw)
                            if flaw:
                                for pb in param_bindings_list_overall[classindex][fsm_no]:
                                    H = pb[0]
                                    H_states = list(H[0])
                                    for h_state in H_states:
                                        if h_state in state:
                                            if pb not in faulty_pb:
                                                faulty_pb.append(pb) # no duplicates
                
                for pb in param_bindings_list_overall[classindex][fsm_no]:
                    if pb not in faulty_pb:
                        pb_per_fsm_fault_removed.append(pb)
                
                                        
                                
                                
                print(str(len(pb_per_fsm_fault_removed)) + "/" + str(len(param_bindings_list_overall[classindex][fsm_no])) + " param retained")
                for pb in pb_per_fsm_fault_removed:
                    print(pb)

                        
                
                pb_per_class_fault_removed.append(pb_per_fsm_fault_removed)
            para_bind_overall_fault_removed.append(pb_per_class_fault_removed)
        return para_bind_overall_fault_removed
    
    def extract_static_pre():
        pass

    def form_pddl(self, para_bind_overall_fault_removed, state_machines_overall_list):
        print(";;********************Learned PDDL domain******************")
        output_file = "output/"+ self.domain_name + "/" +  self.domain_name + ".pddl"
        write_file = open(output_file, 'w')
        write_line = "(define"
        write_line += "  (domain "+ self.domain_name+")\n"
        write_line += "  (:requirements :typing)\n"
        write_line += "  (:types"
        for class_name in self.type_names:
            write_line += " " + class_name
        write_line += ")\n"
        write_line += "  (:predicates\n"

        # one predicate to represent each object state

        predicates = []
        for class_index, pb_per_class in enumerate(para_bind_overall_fault_removed):
            for fsm_no, pbs_per_fsm in enumerate(pb_per_class):
                for state_index, state in enumerate(state_machines_overall_list[class_index][fsm_no].nodes()):
                    
                    state_set = set(state.split('|'))
                    predicate = ""
            
                    write_line += "    (" + self.type_names[class_index] + "_fsm" + str(fsm_no) + "_" +  state
                    predicate += "    (" + self.type_names[class_index] + "_fsm" + str(fsm_no) + "_" + state
                    for pb in pbs_per_fsm:
                            if set(pb[0][0]) <= state_set:
                                if " ?"+pb[1] + " - " + str(pb[0][8]) not in predicate:
                                    write_line += " ?"+pb[1] + " - " + str(pb[0][8])
                                    predicate += " ?"+pb[1] + " - " + str(pb[0][8])
            
                    write_line += ")\n"
                    predicate += ")"
                    predicates.append(predicate)
        write_line += "  )\n"
                    
        for action_index, action in enumerate(self.raw_actions):
            write_line += "\n"
            write_line += "  (:action"
            write_line += "  " + action + " "
            write_line += "  :parameters"
            write_line += "  ("
            arg_already_written_flag = False
            params_per_action = []
            args_per_action = []
            for seq in self.raw_seqs:
                for actarg_tuple in seq:
                    if not arg_already_written_flag:
                        if actarg_tuple[0] == action:
                            arglist = []
                            for arg in actarg_tuple[1]:
                                write_line += "?"+arg + " - " + self.type_names[self.get_type_index(arg)] + " "
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
                    current_class_index = self.get_type_index(arg)
                    for fsm_no, G in enumerate(state_machines_overall_list[current_class_index]):
        #                
                        for start, end, weight in G.edges(data='weight'):
                            _actions = weight.split('|')
                            for _action in _actions:
                                
                                if _action.split('.')[0] == action:
                                    for predicate in predicates:
                                        pred = predicate.split()[0].lstrip("(")
                                        clss = pred.split('_')[0]
                                        fsm = pred.split('_')[1]
                                        state = set(pred.split('_')[2].replace('))',')').split('|'))



                                        if clss == self.type_names[current_class_index]:
                                            if fsm == "fsm" + str(fsm_no):

                                                if state == set(start.split('|')):

                                                    if predicate not in preconditions:
                                                        preconditions.append(predicate)

                                                if state == set(end.split('|')):
                                                    if predicate not in effects:
                                                        effects.append(predicate)
                                    break
                                                
            
                        

            write_line += "   :precondition"
            write_line += "   (and\n"
            for precondition in preconditions:
                # precondition = precondition.replace(?)
                write_line += "    "+precondition+"\n"
            write_line += "   )\n"
            write_line += "   :effect"
            write_line += "   (and\n"
            for effect in effects:
                write_line += "    " + effect + "\n"
            write_line += "  )"

            write_line += ")\n"

        write_line += ")\n" #domain ending bracket


        print(write_line)

        write_file.write(write_line)
        write_file.close()

    def validate_and_fix_pddl(self,para_bind_overall_fault_removed, state_mappings_class, state_machines_overall_list,state_machines_overall_list_2):
        print(";;********************Fixed PDDL domain******************")
        output_file = "output/"+ self.domain_name + "/" +  self.domain_name + ".pddl"
        write_file = open(output_file, 'w')
        write_line = "(define"
        write_line += "  (domain "+ self.domain_name+")\n"
        write_line += "  (:requirements :typing)\n"
        write_line += "  (:types"
        for class_name in self.type_names:
            write_line += " " + class_name
        write_line += ")\n"
        write_line += "  (:predicates\n"

        # one predicate to represent each object state

        predicates = []
        for class_index, pb_per_class in enumerate(para_bind_overall_fault_removed):
            for fsm_no, pbs_per_fsm in enumerate(pb_per_class):
                state_mapping = state_mappings_class[class_index][fsm_no]
                
                for state_index, state in enumerate(state_machines_overall_list[class_index][fsm_no].nodes()):
                    
                    state_set = set(state.split('|'))
                    predicate = ""
            
                    write_line += "    (" + self.type_names[class_index] + "_fsm" + str(fsm_no) + "_state" +  str(state_mapping[state])
                    predicate += "    (" + self.type_names[class_index] + "_fsm" + str(fsm_no) + "_state" + str(state_mapping[state])
                    for pb in pbs_per_fsm:
                            if set(pb[0][0]) <= state_set:
                                if " ?"+pb[1] + " - " + str(pb[0][8]) not in predicate:
                                    write_line += " ?"+pb[1] + " - " + str(pb[0][8])
                                    predicate += " ?"+pb[1] + " - " + str(pb[0][8])
            
                    write_line += ")\n"
                    predicate += ")"
                    predicates.append(predicate)
        write_line += "  )\n"
                    
        for action_index, action in enumerate(self.raw_actions):
            write_line += "  (:action"
            write_line += "  " + action + " "
            write_line += "  :parameters"
            write_line += "  ("
            arg_already_written_flag = False
            params_per_action = []
            args_per_action = []
            for seq in self.raw_seqs:
                for actarg_tuple in seq:
                    if not arg_already_written_flag:
                        if actarg_tuple[0] == action:
                            arglist = []
                            for arg in actarg_tuple[1]:
                                write_line += "?"+arg + " - " + self.type_names[self.get_type_index(arg)] + " "
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
                    current_class_index = self.get_type_index(arg)
                    for fsm_no, G in enumerate(state_machines_overall_list[current_class_index]):
                        G_int = state_machines_overall_list_2[current_class_index][fsm_no]
                        state_mapping = state_mappings_class[current_class_index][fsm_no]
                        for start, end, weight in G_int.edges(data='weight'):
                            _actions = weight.split('|')
                            for _action in _actions:
                                if _action.split('.')[0] == action:
                                    for predicate in predicates:
                                        pred = predicate.split()[0].lstrip("(")
                                        clss = pred.split('_')[0]
                                        fsm = pred.split('_')[1]
                                        state_ind = pred.split('_')[2].rstrip(")")[-1]

                                        if clss == self.type_names[current_class_index]:
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
t.locm()