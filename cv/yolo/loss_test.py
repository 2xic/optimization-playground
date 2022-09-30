
import unittest
from coco2yolo import Coco2Yolo
from loss import yolo_loss
from constants import Constants
from loss import prediction_2_grid

class loss_test(unittest.TestCase):
    def test_1(self):
        coco2Yolo = Coco2Yolo()
        constants = Constants()
        p = [0, ] * constants.CLASSES
        p[4] = 1
        grid_box = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    bounding_boxes=[199.84, 200.46, 77.71, 70.88]
                ))
                + p
            ],
            constants
        )
        grid_box_truth = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    bounding_boxes=[199.84, 200.46, 77.71, 70.88]
                ))

            ],
            constants
        )

        assert yolo_loss(
            grid_box,
            grid_box_truth,
            constants
        ) == 0


    def test_2(self):
        coco2Yolo = Coco2Yolo()
        constants = Constants()

        p = [0, ] * constants.CLASSES
        p[4] = 1

        grid_box = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    # SMALL ADJUSTMENT TO BOUNDING BOX
                    bounding_boxes=[198.84, 18.46, 77.71, 70.88]
                ))
                + p
            ],
            constants
        )
        grid_box_truth = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    bounding_boxes=[199.84, 200.46, 77.71, 70.88]
                ))

            ],
            constants
        )

        assert yolo_loss(
            grid_box,
            grid_box_truth,
            constants
        ) != 0

    def test_3(self):
        coco2Yolo = Coco2Yolo()
        constants = Constants()

        p = [0, ] * constants.CLASSES
        p[5] = 1

        grid_box = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    # SMALL ADJUSTMENT TO BOUNDING BOX
                    bounding_boxes=[198.84, 18.46, 77.71, 70.88]
                ))
                + p
            ],
            constants
        )
        grid_box_truth = prediction_2_grid(
            [
                [4, ] +
                list(coco2Yolo.coco2yolo(
                    width=640,
                    height=247,
                    bounding_boxes=[199.84, 200.46, 77.71, 70.88]
                ))

            ],
            constants
        )

        assert yolo_loss(
            grid_box,
            grid_box_truth,
            constants
        ) != 0

if __name__ == '__main__':
    unittest.main()


