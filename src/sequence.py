class Type:
    raw_name: str

class Argument:
    raw_name: str
    type: Type

class Action:
    raw_name: str
    arg_types= []
    args = []

    args_length = 0

    def __init__(self, action) -> None:
        self.raw_name = action.split("")[0]

class Sequence:
    actions: list[Action] = []





