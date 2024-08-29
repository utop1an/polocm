import random
from pddl_parser.pddl_file import open
from translate import pddl_to_sas
from translate.normalize import normalize
from utils import (
    set_timer_throw_exc,
    TraceSearchTimeOut,
)

class RandomPlanner:
    def __init__(self, domain, problem, plan_len=10, num_traces=1, seed=None, max_time=30):
        # Store only the file paths and minimal configuration
        self.domain = domain
        self.problem = problem
        self.plan_len = plan_len
        self.num_traces = num_traces
        self.seed = seed
        self.max_time = max_time
        self.task = None  # Task is initialized later to avoid non-pickleable issues

    def initialize_task(self):
        """Initialize the task from domain and problem files. This should be called in the worker process."""
        if self.seed:
            random.seed(self.seed)
        # Load and normalize the task only when needed
        task = open(self.domain, self.problem)
        normalize(task)
        self.task = pddl_to_sas(task)

    def generate_traces(self):
        if self.task is None:
            # Make sure the task is initialized in the worker process
            self.initialize_task()

        traces = []
        for _ in range(self.num_traces):
            traces.append(self.generate_single_trace_setup())
        return traces

    def generate_single_trace_setup(self):
        @set_timer_throw_exc(
            num_seconds=self.max_time, exception=TraceSearchTimeOut, max_time=self.max_time
        )
        def generate_single_trace():
            state = tuple(self.task.init.values)
            valid = False

            while not valid:
                trace = []
                for i in range(self.plan_len):
                    ops = self.get_applicable_operators(state)
                    if not ops:  # Simplified empty check
                        break
                    op = random.choice(ops)
                    trace.append(op)
                    state = self.apply_operator(state, op)
                if len(trace) == self.plan_len:
                    valid = True

            return trace

        return generate_single_trace()

    def get_applicable_operators(self, state):
        """Get all operators applicable in the current state."""
        ops = []
        for op in self.task.operators:
            applicable = all(state[var] == pre for var, pre, _, _ in op.pre_post if pre != -1)
            if applicable:
                ops.append(op)
        return ops

    def apply_operator(self, state, op):
        """Apply an operator to a given state and return the new state."""
        new_state = list(state)
        for var, _, post, _ in op.pre_post:
            new_state[var] = post
        return tuple(new_state)
