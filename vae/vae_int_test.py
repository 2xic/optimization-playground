import unittest
from mock_model import MockModel
from optimization_utils.envs.SimpleEnv import SimpleEnv
from epsilon import Epsilon
from statistics import mode
from simple_env_trajectory import play
from torch_model import TorchModel

class VaeInt(unittest.TestCase):
    def test_should_execute(self):
        model = TorchModel()

        (_, lossInteractor, _) = play(model, Epsilon())

        assert 0 < lossInteractor.iterations

