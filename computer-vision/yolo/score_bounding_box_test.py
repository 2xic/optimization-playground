import unittest
from score_bounding_box import get_coordinates_of_tensor
import torch
from constants import Constants


class TestScoreBoundingBox(unittest.TestCase):
    def test_should_execute(self):
        constants = Constants()

        tensor = torch.zeros((
            constants.GRID_SIZE * constants.GRID_SIZE *
            (constants.BOUNDING_BOX_COUNT * 5 + constants.CLASSES)
        ))
        

        output = get_coordinates_of_tensor(tensor, constants, confidence_threshold=-1)

        assert len(output) > 0

    def test_should_filter_confidence(self):
        constants = Constants()

        tensor = torch.zeros((
            constants.GRID_SIZE * constants.GRID_SIZE *
            (constants.BOUNDING_BOX_COUNT * 5 + constants.CLASSES)
        ))
        

        output = get_coordinates_of_tensor(tensor, constants, 1)
        print(output)

        assert len(output) == 0


if __name__ == '__main__':
    unittest.main()
