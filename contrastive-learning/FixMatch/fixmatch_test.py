import unittest

from fixmatch import FixMatch
import torch


class TestFixMatch(unittest.TestCase):


    def test_pseudo_rows_label(self):
        loss = FixMatch()
        loss.adjust_label_size = False
        output, _ = loss.get_psuedo_label(torch.tensor(
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

        assert output[2][0] == 0.1
        assert output[2][1] == 0.3
        assert output[2][2] == 0.2


    def test_pseudo_multiple_rows_label(self):
        loss = FixMatch()
        loss.adjust_label_size = False

        output, _ = loss.get_psuedo_label(torch.tensor(
            [
                [0.8, 0.9, 0.7],
                [0.9, 0.8, 0.7],
                [0.1, 0.3, 0.2],
                [0, 0.8, 0.9],
            ]
        ))
        assert output[0][0] == 0
        assert output[0][1] == 1
        assert output[0][2] == 0

        assert output[1][0] == 1
        assert output[1][1] == 0
        assert output[1][2] == 0

        assert output[2][0] == 0.1
        assert output[2][1] == 0.3
        assert output[2][2] == 0.2

        assert output[3][0] == 0
        assert output[3][1] == 0
        assert output[3][2] == 1

    def test_should_remove_rows_less_than_thresshold(self):
        loss = FixMatch()
        output, _ = loss.get_psuedo_label(torch.tensor(
            [
                [0.8, 0.9, 0.7],
                [0.9, 0.8, 0.7],
                [0.1, 0.3, 0.2],
                [0, 0.8, 0.9],
            ]
        ))
        assert output[0][0] == 0
        assert output[0][1] == 1
        assert output[0][2] == 0

        assert output[1][0] == 1
        assert output[1][1] == 0
        assert output[1][2] == 0

        assert output[2][0] == 0
        assert output[2][1] == 0
        assert output[2][2] == 1

if __name__ == '__main__':
    unittest.main()
