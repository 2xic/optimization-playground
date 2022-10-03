import imp
import unittest
import numpy as np
from mixup import MixUp


class TestMixUp(unittest.TestCase):

    def test_mixup(self):
        x1 = np.random.rand(30, 30)
        x2 = np.random.rand(30, 30)

        y1 = np.asarray([0, 1, 0])
        y2 = np.asarray([0, 0, 1])

        (x, y) = MixUp()(x1, y1, x2, y2)

        assert y[0] == 0
        assert y[1] > 0
        assert y[2] > 0

if __name__ == '__main__':
    unittest.main()

