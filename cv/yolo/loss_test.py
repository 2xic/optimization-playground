
from lib2to3.pytree import convert
import unittest
from coco2yolo import Coco2Yolo
from bounding_box import ImageBoundingBox
from loss import yolo_loss
from score_bounding_box import get_coordinates_of_tensor
from constants import Constants
import torch

class loss_test(unittest.TestCase):
    def test_should_execute(self):
        coco2Yolo = Coco2Yolo()

        convert_results = coco2Yolo.coco2yolo(
            width=640,
            height=247,
            bounding_boxes=[199.84, 200.46, 77.71, 70.88]
        )

        constants = Constants()

        prediction_tesnor = torch.zeros((
            constants.GRID_SIZE * constants.GRID_SIZE *
            (constants.BOUNDING_BOX_COUNT * 5 + constants.CLASSES)
        ))

        assert yolo_loss(
            prediction_tesnor,
            convert_results
        ) is not None



if __name__ == '__main__':
    unittest.main()


