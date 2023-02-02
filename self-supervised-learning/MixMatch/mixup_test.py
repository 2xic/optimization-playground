import unittest
from mixup import MixUp
import torch

class TestMixUp(unittest.TestCase):

    def test_mixup(self):
        x1 = torch.rand(30, 30)
        x2 = torch.rand(30, 30)

        y1 = torch.asarray([0, 1, 0])
        y2 = torch.asarray([0, 0, 1])

        (x, y) = MixUp()(x1, y1, x2, y2, device="cpu")

        assert y[0] == 0
        assert y[1] > 0
        assert y[2] > 0

if __name__ == '__main__':
    unittest.main()

