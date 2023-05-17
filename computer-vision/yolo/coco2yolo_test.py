import unittest
from coco2yolo import Coco2Yolo
from bounding_box import ImageBoundingBox
from constants import Constants
import numpy as np

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

    def test_should_execute_with_file(self):
        coco2Yolo = Coco2Yolo(Constants(), categories=[18])
        coco2Yolo.load_annotations()

        convert_results = coco2Yolo.load("000000049097.jpg")
        img_width, img_height = convert_results["original_size"]
        actual_bounding_box = (
            379.3671875, 230.29085872576178, 453.609375, 435.42936288088646)
        # convert_results["bounding_boxes"][0]
#        print(actual_bounding_box)
#        print(convert_results["yolo_bounding_boxes"][0])

        x_center, width, y_center, height = convert_results["yolo_bounding_boxes"][0]

        convert_back = ImageBoundingBox().convert_yolo_2_coco(
            image_x=img_width,
            image_y=img_height,

            x_center=x_center,
            width=width,
            y_center=y_center,
            height=height
        )
        convert_results = convert_back.bounding_box[0]

     #   print("converted back")
   ##     print(convert_results)
        assert len(convert_results) == len(actual_bounding_box)
        for i in range(len(convert_results)):
            assert np.allclose(
                convert_results[i], actual_bounding_box[i]), f"failed at {i}"


if __name__ == '__main__':
    unittest.main()
