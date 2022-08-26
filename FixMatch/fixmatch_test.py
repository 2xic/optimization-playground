import unittest

from fixmatch import FixMatch
import torch


class TestFixMatch(unittest.TestCase):


    def test_pseudo_rows_label(self):
        loss = FixMatch()
        output = loss.get_psuedo_label(torch.tensor(
            [
                [0.8, 0.9, 0.7],
                [0.9, 0.8, 0.7],
                [0.1, 0.3, 0.2]
            ]
        ))
        assert output[0][0] == 0
        assert output[0][1] == 1
        assert output[0][2] == 0

        assert output[1][0] == 1
        assert output[1][1] == 0
        assert output[1][2] == 0


if __name__ == '__main__':
    unittest.main()
