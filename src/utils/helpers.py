import os
import numpy as np
from tabulate import tabulate
from IPython.display import display, Markdown
import string
import pandas as pd

def complete_PO(PO_matrix):
    for i in range(len(PO_matrix)):
        for j in range(len(PO_matrix)):
            if i==j:
                continue
            current = PO_matrix.iloc[i,j]
            if (not pd.isna(current) and current == 1):

                # complete matrix based on transitivity of PO
                # if a>b, b>c, then a>c
                for x in range(len(PO_matrix)):
                    if x==i or x ==j:
                        continue

                    next = PO_matrix.iloc[j,x]
                    if (next == 1):
                        PO_matrix.iloc[i,x] = 1
                        PO_matrix.iloc[x,i] = 0

def complete_FO(FO_matrix, PO_matrix):
    for i in range(len(PO_matrix)):
        for j in range(len(PO_matrix)):
            if i==j:
                continue
            current_PO = PO_matrix.iloc[i,j]
            if current_PO == 0:
                FO_matrix.iloc[i,j] = 0
            elif current_PO == 1:
                flag=1
                for x in range(len(FO_matrix)):
                    if x != i and x !=j:
                        ix = PO_matrix.iloc[i,x]
                        xj = PO_matrix.iloc[x,j]
                        # not sure
                        if (pd.isna(ix)or pd.isna(xj)):
                            flag =2
                        # FO_ij should be 0
                        if ix==1 and xj==1:
                            flag=0
                            break
                # No change, FO_ij should be 1
                if flag==1:
                    FO_matrix.iloc[i,j]=1
                    FO_matrix.iloc[j,i] = 0
                    # check nans
                    for y in range(len(FO_matrix)):
                        if y!=i and y!=j:
                            FO_matrix.iloc[i,y] = 0
                            FO_matrix.iloc[y,j] = 0
                # FO_ij should be 0
                elif flag == 0:
                    FO_matrix.iloc[i,j]=0


def print_table(matrix):
    display(tabulate(matrix, headers='keys', tablefmt='html'))


def printmd(string):
    display(Markdown(string))

def check_well_formed(df):
    
    
    # Early exit if the matrix is full of zeros
    if (df == 0).all(axis=None):  # Vectorized check for all-zero matrix
        return False
    
    # Convert the DataFrame to a NumPy array for faster operations
    arr = df.to_numpy()

    # Loop over all pairs of rows
    for i in range(arr.shape[0] - 1):
        for j in range(i + 1, arr.shape[0]):
            row1, row2 = arr[i, :], arr[j, :]
            
            # Find the indices where both rows have positive values
            common_cols = (row1 > 0) & (row2 > 0)
            
            if np.any(common_cols):  # If there is at least one common positive value
                # Check if there are "holes" (i.e., one row has a positive value, the other has zero)
                if np.any((row1 > 0) & (row2 == 0)) or np.any((row1 == 0) & (row2 > 0)):
                    return False  # If holes are found, it's not well-formed
    return True  # If no issues are found, it's well-formed

def check_valid(subset_df, valid_pairs):
    arr = subset_df.to_numpy()
    index = subset_df.index
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0:
                # for each pair of transtion <t1,t2> in M
                ordered_pair = (index[i], index[j])
                if ordered_pair not in valid_pairs:
                    return False
    return True