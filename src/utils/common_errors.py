class PercentError(Exception):
    """Raised when the user attempts to supply an invalid percentage of fluents to hide."""

    def __init__(
        self,
        message="The percentage supplied is invalid.",
    ):
        super().__init__(message)

class InvalidModel(Exception):
    """
    Raised when the input model is invalid.
    """

    def __init__(
        self,
        model_name,
        detail="Unknown",
    ):
        dmessage = f"The provided model [{model_name}] is invalid: {detail}"
        super().__init__(message)

class InvalidActionSequence(Exception):
    """
    Raised when the input action sequence is invalid.
    """

    def __init__(
        self,
        message="Unknown",
    ):
        message = f"The provided action sequence is invalid: {message}"
        super().__init__(message)

class InvalidMLPTask(Exception):
    """
    Raised when the MLP task is invalid.
    """

    def __init__(
        self,
        message="Unknown",
        num_vars = 0,
        num_constraints = 0
    ):
        message = f"The provided MLP task with {num_vars} vars and {num_constraints} constraints is not solvable: {message}"
        super().__init__(message)