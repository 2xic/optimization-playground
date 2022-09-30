from collections import defaultdict
import json
from unicodedata import category
from PIL import Image
from typing import List

from helpers import get_local_dir


class Coco2Yolo:
    def __init__(self) -> None:
        pass

    def load_annotations(self):
        self.annotations = open(
            get_local_dir("annotations/instances_train2017.json"), "r").read()
        self.annotations = json.loads(self.annotations)
        self.id_category = {

        }

        for i in self.annotations['categories']:
            self.id_category[i['id']] = i['name']

        self.image_bbox = defaultdict(list)

        for i in self.annotations['annotations'][:10]:
            image_id = str(i['image_id'])

            name = list("000000000000")
            name[-len(image_id):] = image_id
            name = "".join(name) + ".jpg"

            self.image_bbox[name].append({
                'category_id': i['category_id'],
                'bbox': i['bbox']
            })

    def load(self, image_name):
        yolo_bounding_boxes = []

        for i in self.image_bbox[image_name]:
            image = Image.open(image_name)
            (width, height) = image.size
            bounding_boxes = i['bbox']
            category_id = i['category_id']

            yolo_bounding_boxes.append(
                [category_id, ] + self.coco2yolo(
                    width=width,
                    height=height,
                    bounding_boxes=bounding_boxes
                )
            )
        return {
            "yolo_bounding_boxes": yolo_bounding_boxes,
            "image": image_name
        }

    def coco2yolo(self, width, height, bounding_boxes: List[int]):
        x, y, delta_w, delta_h = bounding_boxes

        return (
            (x + delta_w / 2) / width,
            delta_w / 2 / width,

            (y + delta_h / 2) / height,
            delta_h / 2 / height
        )


if __name__ == "__main__":
    example = Coco2Yolo()
    print(json.dumps(example.id_category))
