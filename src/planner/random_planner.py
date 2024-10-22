import random
from pddl_parser.pddl_file import open
from translate import pddl_to_sas
from translate.normalize import normalize
from utils import (
    set_timer_throw_exc,
    TraceSearchTimeOut,
    GeneralTimeOut
)

class RandomPlanner:
    def __init__(self, domain, problem, plan_len=10, num_traces=1, seed=None, max_time=30):
        # Store only the file paths and minimal configuration
        self.domain = domain
        self.problem = problem
        self.plan_len = plan_len
        self.num_traces = num_traces
        
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.max_time = max_time
        
        self.initialize_initial_task()


    @set_timer_throw_exc(num_seconds=30, exception=GeneralTimeOut, max_time=30, source="random planner")
    def initialize_task(self):
        """Initialize the task from domain and problem files. This should be called in the worker process."""
        
        # Load and normalize the task only when needed
        task = open(self.domain, self.problem)
        normalize(task)
        self.task = task
        self.sas_task = pddl_to_sas(task)

    def generate_traces(self):
        if self.sas_task is None:
            # Make sure the task is initialized in the worker process
            self.initialize_task()

        traces = []
        for i in range(self.num_traces):
            trace = self.generate_single_trace_setup()
            trace_with_type_info= self.add_type_info(trace)
            traces.append(trace_with_type_info)
        return traces

    def generate_single_trace_setup(self):
        @set_timer_throw_exc(
            num_seconds=self.max_time, exception=TraceSearchTimeOut, max_time=self.max_time
        )
        def generate_single_trace():

            valid = False
            while not valid:
                state = tuple(self.sas_task.init.values)
                trace = []
                for i in range(self.plan_len):
                    ops = self.get_applicable_operators(state)
                    if not ops or len(ops) == 0:  # Simplified empty check
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
        for op in self.sas_task.operators:
            conditions = op.get_applicability_conditions()
            applicable = all(state[var] == val for var, val in conditions)
            if applicable:
                ops.append(op)
        return ops

    def apply_operator(self, state, op):
        """Apply an operator to a given state and return the new state."""
        new_state = list(state)
        for var, _, post, _ in op.pre_post:
            new_state[var] = post
        return tuple(new_state)
    
  

    def add_type_info(self, trace):
        t = []
        for op in trace:
            action = op.name.strip().strip("()").split(" ")
            action_name = action[0]
            o = self.task.get_action(action_name)
            assert o is not None, f"Action {action_name} not found in domain"

            args = action[1:]
            arg_types = [p.type_name for p in o.parameters]
            arg_with_types = [arg + "?"+ t for arg, t in zip(args, arg_types)]
            new_action = f"({action_name} {' '.join(arg_with_types)})"
            t.append(new_action)
        return t
