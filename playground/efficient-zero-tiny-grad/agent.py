from config import Config
from model import Model
from mcts import MonteCarloSearchTree
from tinygrad import nn

class Agent:
    def __init__(self, config: Config, env) -> None:
        self.config = config
        self.model = Model(self.config)
        self.opt = nn.optim.Adam(nn.state.get_parameters(self.model))
        # TODO: Replay buffer
        self.env = env

    def play(self):
        self.env.reset()
        # Play one epoch 
        self.mcts = MonteCarloSearchTree.from_state(
            self.env.state,
            self.model,
            self.config,
        )
        self.mcts.expand()



    def get_action(self, state):
        # TODO: mcts should use the NN 
        return self.mcts.get_action(state)
