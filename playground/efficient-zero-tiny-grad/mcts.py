"""
The implementation of the monte carlo search tree
"""
from typing import Dict, List
from config import Config
from model import Model
import numpy as np
import random
from tinygrad import Tensor
import numpy as np

class Node:
    def __init__(self, state, reward, parent=None) -> None:
        self.state = state
        self.children: Dict[Node] = {}
        self.parent = parent
        self.visited_count = 0
        self.height = 0 if parent is None else parent.height + 1
        self.reward = reward

    @property
    def q_value(self):
        # TODO: I think this should just be an lstm
        return self.reward

    @property
    def sibling_visited_score(self):
        score = 0
        nodes: List[Node] = self.children.values()
        for i in nodes:
            score += i.visited_count
        return score

class MonteCarloSearchTree:
    def __init__(self, node, model: Model, config: Config) -> None:
        self.root = node
        self.config = config
        self.model = model

    @staticmethod
    def from_state(state, model: Model, config: Config):
        encoded_state = model.encode_state(Tensor(state))
        reward = model.get_state_reward(encoded_state)
        return MonteCarloSearchTree(Node(encoded_state, reward), model, config)
    
    def expand(self):
        for _ in range(self.config.max_iterations):
            node = self.root
            while node.height < self.config.max_depth:
                # TODO: This should be replaced by the model
                index = random.randint(0, self.config.num_actions - 1)
                if index in node.children:
                    node = node.children[index]
                    node.visited_count += 1
                else:
                    next_state = self.model.get_next_state(
                        node.state,
                        index
                    )
                    reward = self.model.get_state_reward(next_state)
                    node.children[index] = Node(
                        next_state,
                        reward,
                        node
                    )
                    node = node.children[index]

    def get_action(self):
        action_score = {}
        for action, node in self.root.children.items():
            node: Node = node
            q_s_a = node.q_value
            p_s_a = self.get_p() * np.sqrt(
                node.sibling_visited_score
            ) / (
                1 + node.visited_count
            )
            score = q_s_a + p_s_a * (
                self.config.c_1 + np.log(
                    (node.sibling_visited_score + self.config.c_2 + 1) / 
                    (self.config.c_2)
                )
            )
            action_score[action] = score
        best_action = max(action_score.keys(), key=lambda x: action_score[x])
        return best_action

    def get_p(self):
        if self.config.is_training:
            return np.random.rand()
        else:
            # TODO: Should use the model 
            raise Exception("Unimplemented")

    # just a debug utility
    def plot(self):
        pass
