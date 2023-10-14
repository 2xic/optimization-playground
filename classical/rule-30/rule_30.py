import numpy as np
from typing import List
from PIL import Image

class Board:
    def __init__(self) -> None:
        self.array = np.zeros((512, 512))

class Rule:
    def __init__(self, pattern, value) -> None:
        self.pattern = pattern
        self.value = value

class RulesLookupTable:
    def __init__(self, rules: List[Rule]) -> None:
        self.values = {}
        for i in rules:
            self.values["".join(list(map(str, i.pattern)))] = i.value

    def apply_rule(self, previous_state, index):
        if 1 < index and (index +  1) < previous_state.shape[0]:
            pattern = "".join(list(map(str, map(int, previous_state[index-2:index+1])))).replace(".", "")
            return self.values[pattern]
        else:
            return 0

rules = RulesLookupTable([
    Rule([1, 1, 1], 0),
    Rule([1, 1, 0], 0),
    Rule([1, 0, 1], 0),
    Rule([1, 0, 0], 1),
    Rule([0, 1, 1], 1),
    Rule([0, 1, 0], 1),
    Rule([0, 0, 1], 1),
    Rule([0, 0, 0], 0),
])

board = Board()
# initial condition
board.array[0, board.array.shape[1] // 2] = 1
for row_index, rows in enumerate(board.array[1:]):
    for index in range(0, rows.shape[0]):
        value = rules.apply_rule(board.array[row_index], index)
        board.array[row_index + 1][index - 1] = value

Image.fromarray(board.array * 255).convert("L").save('rule_30.png')
