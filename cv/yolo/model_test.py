
import unittest
from model import Yolo
import torch

class TestYoloModel(unittest.TestCase):
    def test_should_execute(self):
        model =  Yolo()
        input = torch.zeros((1, 64, 500, 500))
        assert model(input) is not None

if __name__ == '__main__':
    unittest.main()
