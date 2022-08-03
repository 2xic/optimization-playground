import unittest
from mock_model import MockModel

from simple_env_trajectory import play

class TestLoss(unittest.TestCase):
    def test_should_execute(self):
        play(MockModel())


if __name__ == '__main__':
    unittest.main()
