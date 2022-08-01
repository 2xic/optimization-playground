
from torch import maximum
from max_item import MaxItem
from mock_model import MockModel


class Q_valuePlanning:
    def __init__(self, model) -> None:
        self.model = model
        self.b_actions = 2
        self.action_size = 4
        self.depth = 4

    def rollout(self, state):
        maxItem = MaxItem()
        for i in range(self.action_size):
            encoded_state = self.model.transition(state, i)
            maxItem.max(
                self._planning(encoded_state, i, self.depth),
                i
            )
        return maxItem.value        

    def _planning(self, state, action, depth):
        reward, gamma, value, s_t = self.model.core(state, action)

        if depth == 1:
            return reward + gamma * value

        actions = [
            (action, self._q_peek(s_t, action)) for i in range(self.action_size)
        ]
        actions = sorted(actions, key=lambda x: x[1])
        maxItem = MaxItem()
        for (action, score) in actions[:self.b_actions]:
            maxItem.max(
                self._planning(
                    s_t, action, depth - 1
                ),
                None
            )
        return reward + gamma * (1/depth * value + (depth-1)/depth * maxItem.numeric_value)
    
    def _q_peek(self, state, action):
        reward, gamma = self.model.outcome(state, action)

        return reward + gamma * self._value_peek(state)

    def _value_peek(self, state):
        # assert depth == 1
        return self.model.value(state)
