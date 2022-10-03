import unittest
from mock_model import MockModel

from q_value_planning import Q_valuePlanning
import torch

class TestQPlan(unittest.TestCase):
    def test_should_execute(self):
        output = Q_valuePlanning(MockModel())
        assert type(output.rollout(torch.rand(10))) == int

    def test_should_plan(self):
        model = MockModel()
        output = Q_valuePlanning(model)
        assert torch.is_tensor(
            output._planning(
                model.transition(None, None),
                1,
                1
            )
        )
        assert output._planning(
                model.transition(None, None),
                1,
                1
            ).item()

if __name__ == '__main__':
    unittest.main()
