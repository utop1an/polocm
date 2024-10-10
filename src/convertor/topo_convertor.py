
from traces import Trace
from traces.step import Step
from traces.partial_ordered_trace import PartialOrderedTrace, PartialOrderedStep
import pandas as pd
from typing import List
from random import *
import numpy as np
import itertools



class TopoConvertor:
    """
    Given...
    """
    
 
    measurement: callable
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

    def flex(self, cm: pd.DataFrame):
        """
        The flex of a poset, or the coverage of poset:
            flex = 1 - cp/tp
        where cp is the number of comparable pairs of items in the poset, and tp is the total number of pairs.
        1 means totally unordered and 0 means totally ordered.
        """
        cp = cm.notna().sum().sum()
        tp = (len(cm)*(len(cm)-1 ))/ 2
        return 1-(cp/tp)
    
  
    def width(self, cm: pd.DataFrame):
        """
        The length of the largest anti-chain of the poset,
        i.e. the largest subset of the poset such that none of the pairs in it is comparable.
        1 means totally unordered and 0 means totally ordered
        """

        def getSubsets(origin, length):
            return list(itertools.combinations(origin, length))

        def checkAllIncomparable(matrix):
            for i in range(len(matrix)):
                for j in range(i+1, len(matrix)):
                    if not pd.isna(matrix.iloc[i,j]):
                        return False
            return True
        
        poset_list = cm.columns

        for i in range(len(poset_list), 2, -1):
            subsets = getSubsets(poset_list, i)
            for subset in subsets:
                matrix = cm.loc[subset , subset]
                if (checkAllIncomparable(matrix)):
                    return (i/len(poset_list))
        return 0
    
    def completeByTransitivity(self, matrix):
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                current = matrix.iloc[i,j]
                if (not pd.isna(current)):
                    for x in range(1, len(matrix)-1):
                        if (j+x)%len(matrix) == i:
                                continue
                        pivoting = False
                        if (j+x < len(matrix)):
                            pivoting = False
                            next = matrix.iloc[j, j+x]
                        else:
                            pivoting = True
                            pivot = matrix.iloc[(j+x)%len(matrix), j]
                            if (not pd.isna(pivot)):
                                next = not pivot
                            else:
                                next = pivot
                        if (current == next):
                            if (pivoting):
                                if (i>(j+x)%len(matrix)):
                                    matrix.iloc[(j+x)%len(matrix), i] = not current
                                else:
                                    matrix.iloc[i, (j+x)%len(matrix)] = current     
                            else:
                                matrix.iloc[i, j+x] = current
        return matrix

    def getTOComparableMatrix(self, trace: Trace):
        indices = [step for step in trace.steps]
        cm = pd.DataFrame(columns=indices, index=indices)
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                cm.iloc[i,j]= 1
        return cm

    def getPOComparableMatrix(self,input_dod, input_cm: pd.DataFrame):
        def destroy(gap):
            step = int(input_cm.size * gap)
            step = max(step, 2)
            for _ in range(step):
                if len(candidates)==0:
                    break
                x,y = candidates.pop()
                output_cm.iloc[x,y] = np.nan
            return self.measurement(output_cm)

        if (input_dod ==0):
            return input_cm, 0
        if(input_dod == 1):
            output_cm = pd.DataFrame(np.nan, columns=input_cm.columns, index=input_cm.index)
            return input_cm, 1
        candidates = []
        output_cm = input_cm.copy()
        dod = self.measurement(input_cm) 
        for i in range(len(input_cm)):
            for j in range(i+1, len(input_cm)):
                if not pd.isna(input_cm.iloc[i,j]):
                    candidates.append((i,j))
        shuffle(candidates)
        if self.strict:
            flag = True
            gap = input_dod
            while flag:
                dod = destroy(gap)
                output_cm = self.completeByTransitivity(output_cm)
                dod = self.measurement(output_cm)
                if (dod >= input_dod):
                    flag = False
                else:
                    gap = input_dod - dod
                    flag = True

        else:
            dod = destroy(dod)

        return output_cm, dod


    def getPOTrace(self,cm: pd.DataFrame, dod: float):
        to_steps: List[Step] = cm.columns
        po_steps: List[PartialOrderedStep] = [
            PartialOrderedStep(to_step.state, to_step.action, to_step.index, []) for to_step in to_steps
        ]
        for i in range(len(cm)):
            for j in range(i+1, len(cm)):
                if cm.iloc[i,j] == 1:
                    po_steps[i].successors.append(j) 
                elif cm.iloc[i,j] == 0:
                    po_steps[j].successors.append(i)
        if self.reorder:
            shuffle(po_steps)
        
        return PartialOrderedTrace(po_steps, dod, cm)
    
    def convert(self, to_trace, degree_of_disorder):
        to_cm = self.getTOComparableMatrix(to_trace)
        po_cm, dod = self.getPOComparableMatrix(degree_of_disorder, to_cm)
        return self.getPOTrace(po_cm, dod)
    

    



