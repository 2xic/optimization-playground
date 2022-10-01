from lib2to3.pytree import convert
import unittest
from coco2yolo import Coco2Yolo
from bounding_box import ImageBoundingBox
from constants import Constants


class Coco2YoloTest(unittest.TestCase):
    def test_should_execute(self):
        coco2Yolo = Coco2Yolo(Constants())
        convert_results = coco2Yolo.coco2yolo(
            width=640,
            height=247,
            bounding_boxes=[199.84, 200.46, 77.71, 70.88]
        )
        results = (199.84, 200.46, 277.55, 271.34000000000003)

        x_center, width, y_center, height = convert_results

        convert_back = ImageBoundingBox().convert_yolo_2_coco(
            image_x=640,
            image_y=247,
            x_center=x_center, 
            width=width, 
            y_center=y_center, 
            height=height
        )
        convert_results = convert_back.bounding_box[0]

        assert len(convert_results) == len(results)
        for i in range(len(convert_results)):
            assert convert_results[i] == results[i]


if __name__ == '__main__':
    unittest.main()
