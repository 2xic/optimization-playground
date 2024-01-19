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
import graphviz

class Node:
    def __init__(self, state, reward, parent=None) -> None:
        self.id = None
        self.state = state
        self.children: Dict[Node] = {}
        self.parent = parent
        self.p = None
            
        self.visited_count = 0
        self.height = 0 if parent is None else parent.height + 1
        self.reward = reward
        # For debugging only
        self._last_calculated_score = None

    @property
    def q_value(self):
        # TODO: I think this should just be an lstm
        return self.reward.item()

    @property
    def sibling_visited_score(self):
        if self.parent is None:
            return 0
        score = 0
        nodes: List[Node] = self.parent.children.values()
        for i in nodes:
            score += i.visited_count
        return score

    @property
    def visited_relative(self):
        if self.parent is None:
            return 0
        return np.sqrt(
            self.sibling_visited_score
        ) / (
            1 + self.visited_count
        )  
    
    def explored_score(self, config: Config):
        return (
            config.c_1
             + np.log2(
                (self.sibling_visited_score + config.c_2 + 1) / 
                (config.c_2)
            )
        )
    
    # debug util
    @property
    def sibling_visited(self):
        nodes: List[Node] = self.children.values()
        visited = []
        for i in nodes:
            visited.append(i.visited_count)
        return visited
    
    @property
    def visited_probabilities(self):
        values = list(map(lambda x: x.visited_count, self.children.values()))
        return Tensor(values).softmax()
    
class MonteCarloSearchTree:
    def __init__(self, node: Node, model: Model, config: Config) -> None:
        self.root: Node = node
        self.config = config
        self.model = model
        self.root.id = 0
        self.current_node_id = self.root.id + 1

    @staticmethod
    def from_state(state, model: Model, config: Config):
        encoded_state = model.encode_state(Tensor(state))
        reward = model.get_state_reward(encoded_state)
        return MonteCarloSearchTree(Node(encoded_state, reward), model, config)
    
    def expand(self):
        for _ in range(self.config.max_iterations):
           # print("=" * 32)
            node = self.root
            while node.height <= self.config.max_depth:
                # TODO: This should be replaced by the model
                index = self.get_action(node)
                if index in node.children:
                    node = node.children[index]
                else:
                    node = self._add_child_nodes(node)
                    node = node.children[index]
                node.visited_count += 1

    def get_action(self, from_node):
        action_score = {}
        if len(from_node.children) == 0:
            from_node = self._add_child_nodes(from_node)
        for action, node in from_node.children.items():
            node: Node = node
            q_s_a = node.q_value
            p_s_a = node.p * node.visited_relative
            explore_value = node.explored_score(self.config)
            score = q_s_a + p_s_a * explore_value
            # Update it
            node._last_calculated_score = score
            action_score[action] = score
        # print(action_score.values())
        best_score = max(action_score.values())
        best_action = random.sample([
            key for (key, value) in action_score.items() if value == best_score
        ], k=1)[0]
        return best_action
    
    def _add_child_nodes(self, parent_node):
        for action in range(self.config.num_actions):
            if action in parent_node.children:
                continue
            # else create it 
            next_state = self.model.get_next_state(
                parent_node.state,
                action
            )
            reward = self.model.get_state_reward(next_state)
            parent_node.children[action] = Node(
                next_state,
                reward,
                parent_node
            )
            parent_node.children[action].p = self.get_p()
            parent_node.children[action].id = self.current_node_id
            self.current_node_id += 1
        return parent_node

    def get_p(self):
        if self.config.is_training:
            return np.random.rand()
        else:
            # TODO: Should use the model 
            raise Exception("Unimplemented")

    # just a debug utility
    def plot(self):
        dot = graphviz.Digraph(format='png')
        nodes = [
            self.root
        ]
        while len(nodes):
            node = nodes.pop(0)
            explore_value = node.explored_score(self.config)
            dot.node(str(node.id), f"height={node.height} id ={id}, visited_count: {node.visited_count}\n\nq(s, a) = {node.q_value}\nvisited_relative: {node.visited_relative}\nAdjusted {explore_value}\n_last_calculated_score: {node._last_calculated_score}", shape="square")
            for (action, i) in node.children.items():
                dot.edge(str(node.id), str(i.id))
                nodes.append(i)
        dot.render("plot_mcts", cleanup=True)
