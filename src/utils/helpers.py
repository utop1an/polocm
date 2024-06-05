import itertools
import os
from tabulate import tabulate
from IPython.display import display, Markdown
import networkx as nx
import pandas as pd
from ipycytoscape import *

def print_table(matrix):
    display(tabulate(matrix, headers='keys', tablefmt='html'))

    # utils
def findsubsets(S,m):
    return set(itertools.combinations(S, m))
    
def printmd(string):
    display(Markdown(string))

def check_well_formed(
        subset_df
    ):
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
    
def check_valid(
        subset_df,
        consecutive_transitions_per_sort
    ):
    df = subset_df.copy()
    # for particular adjacency matrix's copy, loop over all pairs of rows
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            if df.iloc[i,j] > 0:
                valid_val_flag = False
                ordered_pair = (df.index[i], df.columns[j])
                for ct_list in consecutive_transitions_per_sort:
                    for ct in ct_list:
                        if ordered_pair == ct:
                            valid_val_flag=True
                # if after all iteration ordered pair is not found, mark the subset as invalid.
                if not valid_val_flag:
                    return False
                
    # return True if all ordered pairs found.
    return True