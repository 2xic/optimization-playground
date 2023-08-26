import unittest
import torch
from .agent import Agent
from .config import Config

class TestAgent(unittest.TestCase):
    def test_agent(self):
        config = Config(
            z_size=10,
            image_size=40,
            action_size=5
        )

        observation = torch.zeros((1, 1, config.image_size, config.image_size))
        last_state = torch.zeros((1, config.z_size))
        actions = torch.zeros((1, 1))
        
        agent = Agent(
            config
        )
        action = agent.get_action(
            observation=observation,
            last_state=last_state,
            action=actions

        )
        assert action is not None

        rollout = agent.rollout(
            observation=observation,
            last_state=last_state,
            action=actions
        )
        assert rollout is not None

        state = agent.forward(
            observation=observation,
            last_state=last_state,
            action=actions
        )
        assert state is not None

if __name__ == '__main__':
    unittest.main()
