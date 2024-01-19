from dataclasses import dataclass


@dataclass
class Config:
    # general
    is_training: bool
    num_actions: int
    state_size: int
    # model config
    state_representation_size: int
    # mcts config
    c_1: float
    c_2: float
    max_depth: int
    max_iterations: int

