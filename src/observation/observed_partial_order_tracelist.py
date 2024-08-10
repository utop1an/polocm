from __future__ import annotations
from collections import defaultdict
from collections.abc import MutableSequence
from warnings import warn
from typing import Callable, Dict, List, Type, Set, TYPE_CHECKING
from inspect import cleandoc
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import Observation, ObservedPartialOrderTrace, ObservedTraceList
from traces.action import Action
from traces.fluent import Fluent

# Prevents circular importing
if TYPE_CHECKING:
    from traces import TraceList


class MissingToken(Exception):
    def __init__(self, message=None):
        if message is None:
            message = (
                f"Cannot create ObservationLists from a TraceList without a Token."
            )
        super().__init__(message)


class TokenTypeMismatch(Exception):
    def __init__(self, token, obs_type, message=None):
        if message is None:
            message = (
                "Token type does not match observation tokens."
                f"Token type: {token}"
                f"Observation type: {obs_type}"
            )
        super().__init__(message)


class ObservedPartialOrderTraceList(ObservedTraceList):
    """A sequence of observations.

    A `list`-like object, where each element is a list of `Observation`s.

    Attributes:
        observations (List[List[Observation]]):
            The internal list of lists of `Observation` objects.
        type (Type[Observation]):
            The type (class) of the observations.
    """

    observations: List[List[Observation]]
    type: Type[Observation]

    def __init__(
        self,
        trace_list: TraceList = None,
        Token: Type[Observation] = None,
        observations: List[ObservedPartialOrderTrace] = None,
        **kwargs,
    ):
        if trace_list is not None:
            if not Token and not observations:
                raise MissingToken()

            if Token:
                self.type = Token

            self.observations = []
            self.tokenize(trace_list, **kwargs)

            if observations:
                self.extend(observations)
                # Check that the observations are of the specified token type
                if self.type and type(observations[0][0]) != self.type:
                    raise TokenTypeMismatch(self.type, type(observations[0][0]))
                # If token type was not provided, infer it from the observations
                elif not self.type:
                    self.type = type(observations[0][0])

        elif observations:
            self.observations = observations
            self.type = type(observations[0][0])

        else:
            self.observations = []
            self.type = Observation

    def __getitem__(self, key: int):
        return self.observations[key]

    def __setitem__(self, key: int, value: List[Observation]):
        self.observations[key] = value
        if self.type == Observation:
            self.type = type(value[0])
        elif type(value[0]) != self.type:
            raise TokenTypeMismatch(self.type, type(value[0]))

    def __delitem__(self, key: int):
        del self.observations[key]

    def __iter__(self):
        return iter(self.observations)

    def __len__(self):
        return len(self.observations)

    def insert(self, key: int, value: ObservedPartialOrderTrace):
        self.observations.insert(key, value)
        if self.type == Observation:
            self.type = type(value[0])
        elif type(value[0]) != self.type:
            raise TokenTypeMismatch(self.type, type(value[0]))

    def get_actions(self) -> Set[Action]:
        actions: Set[Action] = set()
        for obs_trace in self:
            for obs in obs_trace:
                action = obs.action
                if action is not None:
                    actions.add(action)
        return actions

    def get_fluents(self) -> Set[Fluent]:
        fluents: Set[Fluent] = set()
        for obs_trace in self:
            for obs in obs_trace:
                if obs.state:
                    fluents.update(list(obs.state.keys()))
        return fluents

    def tokenize(self, trace_list: TraceList, **kwargs):
        for trace in trace_list:
            tokens = trace.tokenize(self.type, **kwargs)
            self.append(tokens)

    
