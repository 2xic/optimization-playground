from typing import List
from helpers import get_local_dir
import json
import matplotlib.pyplot as plt
from PIL import Image

class KeyPoint:
    def __init__(self, x, y, visible) -> None:
        #          https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        self.x = x
        self.y = y
        self.visible = visible

    def is_visible(self):
        return self.visible == 2

    def __str__(self):
        return f"({self.x}, {self.y}, {self.visible})"

    def __repr__(self) -> str:
        return self.__str__()

class ImageLabel:
    def __init__(self, image_id, keypoints: List[KeyPoint]) -> None:
        self.image_id = image_id
        self.keypoints = keypoints

        self.name = list("000000000000")
        self.name[-len(image_id):] = image_id
        self.name = "".join(self.name) + ".jpg"

        self.image = Image.open(
            get_local_dir("train2017/" + self.name)
        )

    def show(self, skeleton=[]):
        for i in self.keypoints:
            plt.scatter(i.x, i.y)

        for i in skeleton:
            point_1 = self.keypoints[i[0] - 1]
            point_2 = self.keypoints[i[1] - 1]

            if point_1.is_visible() and point_2.is_visible():
                x_values = [point_1.x, point_2.x]
                y_values = [point_1.y, point_2.y]

                plt.plot(x_values, y_values, 'bo', linestyle="--")

        plt.imshow(self.image)
        plt.show()


class Coco:
    def __init__(self) -> None:
        self.results = []


    def load_annotations(self):
        self.annotations = open(
            get_local_dir("annotations/person_keypoints_train2017.json"), "r").read()
        self.annotations = json.loads(self.annotations)
        self.images = {}
        for entry in self.annotations['annotations'][:1]:
            keypoints = entry['keypoints']
            results = []
            for i in range(0, len(keypoints), 3):
                results.append(KeyPoint(*keypoints[i:i+3]))
            self.results.append(
                ImageLabel(
                    image_id=str(entry['image_id']),
                    keypoints=results
                )
            )
        self.skeleton = self.annotations['categories'][0]['skeleton']

        return self

    def show(self):
        self.results[0].show(self.skeleton)
        
