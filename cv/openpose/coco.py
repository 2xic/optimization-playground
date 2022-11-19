from typing import List
from PIL import Image
import torch
from helpers import get_local_dir
import json
import matplotlib.pyplot as plt

class KeyPoint:
    def __init__(self, x, y, visible) -> None:
        #          https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        self.x = x
        self.y = y
        self.visible = visible

    def is_visible(self):
        return self.visible == 2

    @property
    def location(self):
#        return torch.tensor([self.x, self.y]).float()
        return torch.tensor([self.y, self.x]).float()

    def __str__(self):
        return f"({self.x}, {self.y}, {self.visible})"

    def __repr__(self) -> str:
        return self.__str__()

class ImageLabel:
    def __init__(self, image_id, keypoints: List[KeyPoint], bbox) -> None:
        self.image_id = image_id
        self.keypoints = keypoints
        self.bbox = bbox

        self.name = list("000000000000")
        self.name[-len(image_id):] = image_id
        self.name = "".join(self.name) + ".jpg"

        self.image = Image.open(
            get_local_dir("train2017/" + self.name)
        )
        self.shape = self.image.size

    def imshow(self, skeleton=[]):
        plt.clf()
        for _, i in enumerate(self.keypoints):
            plt.scatter(i.x, i.y)

        for _, i in enumerate(skeleton):
            point_1 = self.keypoints[i[0] - 1]
            point_2 = self.keypoints[i[1] - 1]

            if point_1.is_visible() and point_2.is_visible():
                x_values = [point_1.x, point_2.x]
                y_values = [point_1.y, point_2.y]

                plt.plot(x_values, y_values, 'bo', linestyle="--")
        plt.imshow(self.image)

    def plot_image_skeleton_keypoints(self, score_coordinate, draw_skeleton=True):
        plt.clf()
        for score, from_c, to_c in score_coordinate:
            if score is not None:
                if draw_skeleton:
                    x_val = [from_c[1], to_c[1]]
                    y_val = [from_c[0], to_c[0]]
                    plt.plot(x_val, y_val, 'bo', linestyle="--")
                else:
                    plt.scatter(from_c[1], from_c[0], color='b')
                    plt.scatter(to_c[1], to_c[0], color='b')
        plt.imshow(self.image)
        plt.show()

    def show(self, skeleton=[]):
        self.imshow(skeleton)
        plt.show()

class Coco:
    def __init__(self) -> None:
        self.results = []
        self.skeleton = None

    def load_annotations(self):
        self.annotations = open(
            get_local_dir("annotations/person_keypoints_train2017.json"), "r").read()
        self.annotations = json.loads(self.annotations)
        self.images = {}
        for entry in self.annotations['annotations'][:10]:
            keypoints = entry['keypoints']
            bbox = entry['bbox']
            results = []
            for i in range(0, len(keypoints), 3):
                results.append(KeyPoint(*keypoints[i:i+3]))
            self.results.append(
                ImageLabel(
                    image_id=str(entry['image_id']),
                    keypoints=results,
                    bbox=bbox,
                )
            )
        self.skeleton = self.annotations['categories'][0]['skeleton']

        return self

    def show(self, index=0):
        self.results[index].show(self.skeleton)
        
    def get_metadata(self, index):
        return {
            "skeleton": self.skeleton,
            "keypoints": self.results[index].keypoints,
            "bbox": self.results[index].bbox,
            "path":  get_local_dir("train2017/" + self.results[index].name),
            "shape": (480, 640)
        }

if __name__  == "__main__":
    obj = Coco()
    obj.load_annotations()
    # obj.show(4)
#    obj.show(6)
  
    
#    print(obj.skeleton)
#    print(obj.results[0].keypoints)
#    print(obj.results[0].name)
