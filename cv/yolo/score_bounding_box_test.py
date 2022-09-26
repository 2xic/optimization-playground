import unittest
from score_bounding_box import get_coordinates_of_tensor
import torch


class TestScoreBoundingBox(unittest.TestCase):
    def test_should_execute(self):
        GRID_SIZE = 7
        BOUNDING_BOX_COUNT = 2
        CLASSES = 20

        tensor = torch.zeros((
            GRID_SIZE * GRID_SIZE * (BOUNDING_BOX_COUNT * 5 + CLASSES)
        ))
        output = get_coordinates_of_tensor(tensor,
                                           GRID_SIZE,
                                           BOUNDING_BOX_COUNT,
                                           CLASSES
        )

        assert output is not None


if __name__ == '__main__':
    unittest.main()
