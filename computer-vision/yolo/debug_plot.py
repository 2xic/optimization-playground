import unittest
from coco2yolo import Coco2Yolo
from bounding_box import ImageBoundingBox
from constants import Constants
import numpy as np

def test_should_execute_with_file():
    constants = Constants()
    coco2Yolo = Coco2Yolo(constants, categories=[58])
    coco2Yolo.load_annotations()


    image = list(coco2Yolo.image_bbox.keys())[0]
    convert_results = coco2Yolo.load(image)
    img_width, img_height = convert_results["original_size"]
    x_center, width, y_center, height = convert_results["yolo_bounding_boxes"][0]
    
    ImageBoundingBox().load_image(
        image,
        constants,        
    ).convert_yolo_2_coco(
        image_x=img_width,
        image_y=img_height,

        x_center=x_center,
        width=width,
        y_center=y_center,
        height=height,

        confidence=1,
    ).save('test_coco_2_yolo.png')

    ImageBoundingBox().load_image(
        image,
        constants,        
    ).load_bbox(
        convert_results["yolo_bounding_boxes"][0],
        confidence=1
    ).save('test_coco_2_yolo_v2.png')

    print(convert_results["raw_bounding_boxes"])
    print(convert_results['classes'])
    print(convert_results['category_names'])
    ImageBoundingBox().load_original_image(image).save_with_raw_coco(
        convert_results["raw_bounding_boxes"],
        name='test_coco_bbox.png'
    )

    print(convert_results["raw_bounding_boxes"])
    print(convert_results['classes'])
    print(convert_results['category_names'])
    ImageBoundingBox().load_image(image, constants).save_with_raw_coco(
        convert_results["bounding_boxes"],
        name='test_coco_bbox_resized.png'
    )

if __name__ == "__main__":
    test_should_execute_with_file()
