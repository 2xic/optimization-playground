from collections import defaultdict
import json
from unicodedata import category
from unittest import result
from PIL import Image
from typing import List
from torchvision import transforms
from constants import Constants
from helpers import convert_image, get_local_dir


class Coco2Yolo:
    def __init__(self, constants: Constants) -> None:
        self.constants = constants

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
        return self

    def iter(self):
        for i in self.image_bbox:
            yield self.load(i)

    def load(self, image_name):
        yolo_bounding_boxes = []
        classes = []

        if image_name not in self.image_bbox:
            print(self.image_bbox.keys())
            raise Exception("image not in image bbox")

        for i in self.image_bbox[image_name]:
            result = convert_image(image_name, self.constants)

            (width, height) = result.size
            bounding_boxes = i['bbox']
            category_id = i['category_id']

            yolo_bounding_boxes.append(
                list(self.coco2yolo(
                    width=width,
                    height=height,
                    bounding_boxes=bounding_boxes
                ))
            )
            classes.append(
                category_id
            )
        image = transforms.ToTensor()(
            convert_image(image_name, self.constants)
        )
        return {
            "name": image_name,
            "path": get_local_dir("train2017/" + image_name),
            "yolo_bounding_boxes": yolo_bounding_boxes,
            "classes": classes,
            "image": image
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
