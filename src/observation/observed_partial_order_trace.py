from typing import List
from . import PartialOrderedActionObservation

class ObservedPartialOrderTrace:

    obs: List[PartialOrderedActionObservation]

    def __init__(self, obs) -> None:
        self.obs = obs
        
    def __getitem__(self, key:int):
        for ob in self.obs:
            if ob.index == key:
                return ob
        return None
    
    def __iter__(self):
        return iter(self.obs)