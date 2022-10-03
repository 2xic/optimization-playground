import unittest
from mock_model import MockModel

from simple_env_trajectory import play
from epsilon import Epsilon

class TestLoss(unittest.TestCase):
    def test_should_execute(self):
        play(MockModel(), Epsilon())


if __name__ == '__main__':
    unittest.main()
