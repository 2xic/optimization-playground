import imp
import unittest
import numpy as np
from sharpen import sharpen
import torch

class TestSharpen(unittest.TestCase):

    def test_sharpen(self):
        x = torch.tensor([0.2, 0.8, 0])
        y = sharpen(x, T=0.1)

        assert self._close(y[0], 0.0)
        assert self._close(y[1], 1.0)
        assert self._close(y[2], 0.0)

        y = sharpen(x, T=0.4)

        assert not self._close(y[0], 0.0)
        assert not self._close(y[1], 1.0)
        assert self._close(y[2], 0.0)


    def _close(self, x, y, eps=1e-4):
        return (max(x, y) - min(x, y)) < eps

if __name__ == '__main__':
    unittest.main()

