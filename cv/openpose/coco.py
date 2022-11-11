from typing import List
from helpers import get_local_dir
import json
import matplotlib.pyplot as plt
from PIL import Image
from confience_map import ConfidenceMap
import torch
from parity_fields import ParityFields

class KeyPoint:
    def __init__(self, x, y, visible) -> None:
        #          https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        self.x = x
        self.y = y
        self.visible = visible

    def is_visible(self):
        return self.visible == 2

    @property
    def locaiton(self):
        return torch.tensor([self.x, self.y]).float()

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
        self.shape = self.image.size

    def imshow(self, skeleton=[]):
        for index, i in enumerate(self.keypoints):
            plt.scatter(i.x, i.y)
            print(index, i.x, i.y)

        for _, i in enumerate(skeleton):
            point_1 = self.keypoints[i[0] - 1]
            point_2 = self.keypoints[i[1] - 1]

            if point_1.is_visible() and point_2.is_visible():
                x_values = [point_1.x, point_2.x]
                y_values = [point_1.y, point_2.y]

                plt.plot(x_values, y_values, 'bo', linestyle="--")

        plt.imshow(self.image)

    def show(self, skeleton=[]):
        self.imshow(skeleton)
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
        
    def plot_confidence(self):
        confidence = ConfidenceMap()
        img_shape = self.results[0].shape
        x = torch.zeros(img_shape)
        current_keypoint = self.results[0].keypoints[5].locaiton.reshape((1, 1, 2))

        i, j = torch.meshgrid(
            torch.arange(img_shape[0]), 
            torch.arange(img_shape[1]), 
            indexing='ij'
        )
        grid_tenosr = torch.dstack([i, j]).float()
        print(grid_tenosr)
        print(grid_tenosr.shape)
        print(current_keypoint.shape)
       # exit(0)

        x = confidence.function(
            grid_tenosr,
            current_keypoint
        )
       # print(x.shape)
      #  print(x)
       # exit(0)
        plt.imshow(x * 255)
        plt.show()

    def plot_paf(self):
        img_shape = self.results[0].shape
        x = ParityFields()
        p_1 = torch.tensor((355, 367)).float()
        p_2 = torch.tensor((423, 314)).float()

        break_now = False
        for i in range(img_shape[0]):
            if break_now:
                break
            for j in range(img_shape[1]):
                res = x.function(i, j, p_1, p_2)
                if torch.is_tensor(res):
                   # plt.scatter(i, j)
                    plt.quiver(p_1[0], p_1[1], (p_2[0] - p_1[0]), (p_2[1] - p_1[0]), angles='xy', scale_units='xy')#, scale=100)
                    plt.quiver(p_1[0], p_1[1], res[0], res[1], angles='xy', scale_units='xy')#, scale=100)
                 #   break_now = True
                    break
                
     #   self.results[0].imshow()
        plt.show()

