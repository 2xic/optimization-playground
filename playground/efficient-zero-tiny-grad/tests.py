import unittest
from env.simple_rl_env import SimpleRlEnv
from config import Config
from agent import Agent
from mcts import MonteCarloSearchTree

class TestEnvironments(unittest.TestCase):
    def test_upper(self):
        a = SimpleRlEnv()
        assert a.step(0)[1] == 1 
        assert a.step(0)[1] == 0
        assert a.step(0)[1] == 1, "This should now be zero ..."
        assert a.step(1)[1] == 1    
        # Winner

class ModelForward(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            is_training=True,
            num_actions=2,
            state_size=2,
            state_representation_size=4,
            max_iterations=10,
            max_depth=0,
            # numbers appendix 3
            c_1=1.25, 
            c_2=19652,
        )

    def test_simple_env_roll_forward(self):
        self.config.max_depth = 0
        env = SimpleRlEnv()
        agent = Agent(self.config, env)
        mcts = MonteCarloSearchTree.from_state(
            env.state,
            agent.model,
            agent.config,
        )
        mcts.expand()
        assert len(mcts.root.children) == 2
        assert mcts.root.sibling_visited[0] != sum(mcts.root.sibling_visited), "Visited only first node"

    def test_longer_env_roll_forward(self):
        self.config.max_iterations = 5_00
        self.config.max_depth = 1
        env = SimpleRlEnv()
        agent = Agent(self.config, env)
        mcts = MonteCarloSearchTree.from_state(
            env.state,
            agent.model,
            agent.config,
        )
        mcts.expand()
        mcts.plot()
        assert len(mcts.root.children) == 2
        assert mcts.root.sibling_visited[0] != sum(mcts.root.sibling_visited), "Visited only first node"
        for i in mcts.root.children.values():
            assert 1 < i.visited_count, f"No visits :("

if __name__ == '__main__':
    unittest.main()
