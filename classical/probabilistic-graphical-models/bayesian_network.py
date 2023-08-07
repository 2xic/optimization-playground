
"""
Figure 14.2 from Artificial intelligence a modern approach version 3

Need to think more about the abstractions for this
"""
import json

class ConditionalProbabilityTable:
    def __init__(self, variables) -> None:
        self.variables = variables
        self.table = {}

    def add_probability(self, state, value):
        variable_entry = "_".join(self.variables)
        state_entry = "_".join(list(map(str, state)))
        revert_state_entry = "_".join(list(map(lambda x: str(not x), state)))

        # if it is a binary variable
        if len(self.variables) == 1:
            self.table[
                variable_entry + "_" + revert_state_entry
            ] = (1 - value)

        self.table[
            variable_entry + "_" + state_entry
        ] = value

        return self
    
    def get_probability(self, state):
        variable_entry = "_".join(self.variables)
        state_entry = "_".join(list(map(str, state)))

        return self.table[
            variable_entry + "_" + state_entry
        ] 
    
    def __str__(self) -> str:
        return json.dumps( self.table, indent=4)

    def __repr__(self) -> str:
        return self.__str__()

class Node:
    def __init__(self) -> None:
        self.cpt = None
        self.parents = None

    def add_probability(self, node):
        self.cpt = node
        return self
    
    def get_probability(self, state):
        return self.cpt.get_probability(state)

burglary = Node().add_probability(
    ConditionalProbabilityTable(
        "B"
    ).add_probability(
        [True],
        0.001
    )
)
earthquake = Node().add_probability(
    ConditionalProbabilityTable(
        "E"
    ).add_probability(
        [True],
        0.02
    )
)

alarm = Node().add_probability(
    ConditionalProbabilityTable(
        ["B", "E"],
    ).add_probability(
        [True, True],
        0.95
    ).add_probability(
        [True, False],
        0.94
    ).add_probability(
        [False, True],
        0.29
    ).add_probability(
        [False, False],
        0.001
    )
)
alarm.parents = [burglary, earthquake]

john_calls = Node().add_probability(
    ConditionalProbabilityTable(
        ["j"],
    ).add_probability(
        [True],
        0.90
    ).add_probability(
        [False],
        0.05
    )
)
john_calls.parents = [alarm]

marry_calls = Node().add_probability(
    ConditionalProbabilityTable(
        ["m"],
    ).add_probability(
        [True],
        0.70
    ).add_probability(
        [False],
        0.01
    )
)
marry_calls.parents = [alarm]

print(alarm.cpt)
print(marry_calls.cpt)



"""
Probability of alarm has sounded, but there is neither a 
burglary nor an earthquake and both John and Mary called.

P(j, m, a, not(b), not(e)) = P(j|a) * P(m|a)*P(a| not (b) and not(e)) * P(not(b)) * P(not(e))
= 0.9 * 0.7 * 0.001*0.999*0.998 = 0.000628
"""

print(marry_calls.get_probability([True]))
print(john_calls.get_probability([True]))
print(alarm.get_probability([False, False]))
print(burglary.get_probability([False]))
print(earthquake.get_probability([False]))

