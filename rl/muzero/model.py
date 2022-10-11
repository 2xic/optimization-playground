from optimization_utils.tree.State import State
import random
import torch

class Model:
    def __init__(self) -> None:
        pass

    def get_action(self, state: State):
        return random.sample(state.possible_actions)

    def get_state_representation(self, state: torch.Tensor):
        return state
