
from traces.trace import Trace
from traces.step import Step
from traces.partial_ordered_trace import PartialOrderedTrace, PartialOrderedStep
import pandas as pd
from typing import Callable, List
from random import *
import numpy as np
import itertools
import math



class TopoConvertor:
    """
    Given...
    """
    
    measurement: Callable
    batch_destroy: bool
    strict: bool
    reorder: bool
    seed: int
    
    def __init__(self, measurement:str='flex', strict = True, reorder = True, rand_seed: int= 42):
       
        if (measurement == 'width'):
            self.measurement = self.width
        else:
            self.measurement = self.flex
        self.strict = strict
        self.reorder = reorder
        self.seed = rand_seed
        if self.seed:
            seed(self.seed)

    def flex(self, cm):
        """
        The flex of a poset, or the coverage of poset:
            flex = 1 - cp/tp
        where cp is the number of comparable pairs of items in the poset, and tp is the total number of pairs.
        1 means totally unordered and 0 means totally ordered.
        """
        total_pairs = cm.shape[0] * (cm.shape[0] - 1) / 2
        comparable_pairs = np.count_nonzero(cm == 1)
        return 1 - comparable_pairs / total_pairs
    
  
    def width(self, cm):
        """
        The length of the largest anti-chain of the poset,
        i.e. the largest subset of the poset such that none of the pairs in it is comparable.
        1 means totally unordered and 0 means totally ordered
        """
        def getSubsets(origin, length):
            return list(itertools.combinations(origin, length))

        def checkAllIncomparable(matrix):
            return np.isnan(submatrix).all()
        
        n = cm.shape[0]
        poset_list = np.arange(n)
        for subset_size in range(n, 1, -1):
            subsets = getSubsets(poset_list, subset_size)
            for subset in subsets:
                submatrix = cm[np.ix_(subset, subset)]
                if checkAllIncomparable(submatrix):
                    return subset_size / n
        return 0
    
    def completeByTransitivity(self, matrix: np.ndarray):
        """
        This function completes a partially ordered matrix by transitivity.
        It ensures that if A < B and B < C, then A < C.
        
        Args:
            matrix: A NumPy 2D array where np.nan represents incomparable pairs, and 1 represents comparable pairs.
        
        Returns:
            A NumPy 2D array with transitive relations completed.
        """

        n = matrix.shape[0]
        
        # Loop over each pair (i, j) in the upper triangle of the matrix
        for i in range(n):
            for j in range(i + 1, n):
                current = matrix[i, j]
                if current == 1:  # If the pair is comparable
                    # Complete the transitivity for all elements between i and j
                    for k in range(j + 1, n):  # Iterate over all elements after j
                        next_val = matrix[j, k]  # Check the relation between j and k
                        if next_val == 1:  # If the relation holds, propagate it
                            matrix[i, k] = 1                  
        return matrix

    def getTOComparableMatrix(self, trace):
        n = len(trace.steps)
        
        # Initialize a matrix of size n x n filled with np.nan
        cm = np.full((n, n), np.nan)
        
        # Set the upper triangular part (excluding the diagonal) to 1 (comparable pairs)
        for i in range(n):
            for j in range(i + 1, n):
                cm[i, j] = 1
        return cm

    def getPOComparableMatrix(self,input_dod, input_cm):
        def destroy(gap, repeats):
            min_step = math.ceil(repeats/10)
            candidates = np.argwhere(output_cm == 1)
            np.random.shuffle(candidates)
            n = input_cm.shape[0]
            step = int(n*(n-1)/2 * gap)
            min_step = min(len(candidates), min_step)
            step = max(step,  min_step)
            idx_to_remove = np.random.choice(len(candidates), size=step, replace = False)
            for idx in idx_to_remove:
                x,y = candidates[idx]  
                output_cm[x,y] = np.nan
        if (input_dod ==0):
            return input_cm, 0
        if(input_dod == 1):
            output_cm = np.full_like(input_cm, np.nan)
            return output_cm, 1
        
        output_cm = input_cm.copy()


        if self.strict:
            repeats=0
            flag = True
            gap = input_dod
            while flag:
                dod = destroy(gap, repeats)
                output_cm = self.completeByTransitivity(output_cm)
                dod = self.measurement(output_cm)
                repeats+=1
                if (dod >= input_dod):
                    flag = False
                else:
                    gap = input_dod - dod
                    flag = True

        else:
            dod = destroy(input_dod, 1)

        return output_cm, dod


    def getPOTrace(self,cm, dod, to_trace: Trace):
        po_steps: List[PartialOrderedStep] = [
            PartialOrderedStep(to_step.state, to_step.action, to_step.index, []) for to_step in to_trace.steps
        ]
        for i in range(len(cm)):
            for j in range(i+1, len(cm)):
                if cm[i,j] == 1:
                    po_steps[i].successors.append(j) 
        if self.reorder:
            shuffle(po_steps)
        
        return PartialOrderedTrace(po_steps, dod)
    
    def convert(self, to_trace, input_dod):
        to_cm = self.getTOComparableMatrix(to_trace)
        po_cm, output_dod = self.getPOComparableMatrix(input_dod, to_cm)
        return self.getPOTrace(po_cm, output_dod, to_trace)
    

    



