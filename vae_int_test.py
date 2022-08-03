import unittest
from mock_model import MockModel
from simple_env import SimpleEnv
        
from statistics import mode
from simple_env_trajectory import play
from torch_model import TorchModel

class VaeInt(unittest.TestCase):
    def test_should_execute(self):
        model = TorchModel()

        (_, lossInteractor) = play(model)

        assert 0 < lossInteractor.iterations

