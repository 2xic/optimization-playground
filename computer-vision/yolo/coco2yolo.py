from collections import defaultdict
import json
from typing import List
from torchvision import transforms
from constants import Constants
from helpers import convert_image, get_local_dir, get_original_image_size


class Coco2Yolo:
    def __init__(self, constants: Constants, categories=[]) -> None:
        self.constants = constants
        """
        {58: 5, 18: 176, 64: 206, 72: 80, 16: 78, 17: 214, 19: 72, 20: 23, 21: 73, 44: 1187, 63: 28, 62: 432, 67: 164, 2: 575, 3: 913, 4: 581, 5: 19, 6: 118, 7: 22, 9: 67, 1: 2775, 13: 18, 28: 127, 32: 49, 37: 11, 54: 12, 65: 23, 77: 75, 82: 204, 85: 158, 90: 103, 8: 80, 10: 177, 11: 14, 14: 16, 15: 102, 22: 21, 25: 6, 34: 12, 35: 33, 36: 19, 38: 30, 39: 1, 40: 1, 41: 46, 42: 58, 43: 10, 46: 89, 47: 338, 48: 38, 49: 214, 50: 107}
        """
        self.categories = categories

    def load_annotations(self, samples=100_000):
        self.annotations = open(
            get_local_dir("annotations/instances_train2017.json"), "r").read()
        self.annotations = json.loads(self.annotations)
        self.id_category = {}

        for i in self.annotations['categories']:
            self.id_category[i['id']] = i['name']
        print(self.id_category)

        self.image_bbox = defaultdict(list)
        classes_mapping = {}
        class_counts = {}

        for i in self.annotations['annotations']: #[:10_000]:
            image_id = str(i['image_id'])

            name = list("000000000000")
            name[-len(image_id):] = image_id
            name = "".join(name) + ".jpg"

            category_id = i['category_id']
            class_counts[category_id] = class_counts.get(category_id, 0) + 1

            if category_id in self.categories and len(self.image_bbox) < samples:
                classes_mapping[category_id] = classes_mapping.get(
                    category_id, 0) + 1
                self.image_bbox[name].append({
                    'category_id': category_id,
                    'category_name': self.id_category[category_id],
                    'bbox': i['bbox']
                })
            if samples <= len(self.image_bbox):
                break
        print(class_counts)
        print(len(self.image_bbox))
        return self

    def iter(self):
        for i in self.get_list():
            yield self.load(i)

    def get_list(self):
        return list(sorted(self.image_bbox.keys()))

    def load(self, image_name):
        yolo_bounding_boxes = []
        classes_mapping = {}
        classes = []
        all_bounding_boxes = []
        raw_bounding_boxes = []

        if image_name not in self.image_bbox:
            print(self.image_bbox.keys())
            raise Exception("image not in image bbox")

        original_size = get_original_image_size(image_name)
        result = convert_image(image_name, self.constants)
        (width, height) = result.size
        # print(("size", original_size))
        # print(("size", (width, height)))
        category_names = []
        for i in self.image_bbox[image_name][:self.constants.BOUNDING_BOX_COUNT]:
            raw_bbox = i['bbox']
            bounding_boxes = list(raw_bbox) + []
            # print(bounding_boxes)
            """
            delta_w = width / original_size[0]
            delta_h = height / original_size[1]

            bounding_boxes[0] *= delta_w
            bounding_boxes[1] *= delta_h
            bounding_boxes[2] *= delta_w
            bounding_boxes[3] *= delta_h
            """
            category_id = i['category_id']

            classes_mapping[category_id] = len(classes)
            yolo_bounding_boxes.append(
                list(self.coco2yolo(
                    width=self.constants.image_width,
                    height=self.constants.image_height,
                    bounding_boxes=bounding_boxes
                ))
            )
            all_bounding_boxes.append(bounding_boxes)
            raw_bounding_boxes.append(raw_bbox)
            classes.append(category_id)
            category_names.append(i['category_name'])
        image = transforms.ToTensor()(
            convert_image(image_name, self.constants)
        )
        return {
            "name": image_name,
            "original_size": [width, height],
            "path": get_local_dir("train2017/" + image_name),
            "yolo_bounding_boxes": yolo_bounding_boxes,
            "bounding_boxes": all_bounding_boxes,
            "raw_bounding_boxes": raw_bounding_boxes,
            'category_names': category_names,
            "classes": classes,
            "image": image
        }

    """
    def coco2yolo(self, width, height, bounding_boxes: List[float]):
        x_min, y_min, box_width, box_height = bounding_boxes

        x_center = (x_min + box_width / 2) / width
        y_center = (y_min + box_height / 2) / height

        x_normalized = box_width / width
        y_normalized = box_height / height
        
        return x_center, x_normalized, y_center,  y_normalized
    """

    def coco2yolo(self, width, height, bounding_boxes: List[float]):
        x_min, y_min, delta_w, delta_h = bounding_boxes
        # print((width, height))
        return (
            (x_min + delta_w / 2) / width,
            delta_w / 2 / width,

            (y_min + delta_h / 2) / height,
            delta_h / 2 / height
        )


if __name__ == "__main__":
    example = Coco2Yolo()
    print(json.dumps(example.id_category))
