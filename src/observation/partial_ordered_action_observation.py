from traces.partial_ordered_step import PartialOrderedStep
from . import InvalidQueryParameter, Observation


class PartialOrderedActionObservation(Observation):
    """The Action Sequence Observability Token.
    Only stores the ordered action sequence, dropping all stateful information.
    For use in LOCM suite algorithms.
    """

    def __init__(
        self,
        step: PartialOrderedStep,
        **kwargs,
    ):
        """
        Creates an ActionObservation object, storing the step.
        Args:
            step (Step):
                The step associated with this observation.
        """

        Observation.__init__(self, index=step.index)

        self.state = None # stateless representation
        self.action = None if step.action is None else step.action.clone()
        self.successors = None if step.successors is None else step.successors.copy()

    def __eq__(self, other):
        return (
            isinstance(other, PartialOrderedActionObservation)
            and self.state == None  #
            and self.action == other.action
            and self.successors == other.successors
        )

    def _matches(self, key: str, value: str):
        if key == "action":
            if self.action is None:
                return value is None
            return self.action.details() == value
        elif key == "fluent_holds":
            if self.state is None:
                return value is None
            return self.state.holds(value)
        else:
            raise InvalidQueryParameter(PartialOrderedActionObservation, key)