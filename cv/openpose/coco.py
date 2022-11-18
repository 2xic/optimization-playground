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
            #print(index, i.x, i.y)

        for _, i in enumerate(skeleton):
            point_1 = self.keypoints[i[0] - 1]
            point_2 = self.keypoints[i[1] - 1]

            if point_1.is_visible() and point_2.is_visible():
                x_values = [point_1.x, point_2.x]
                y_values = [point_1.y, point_2.y]

                plt.plot(x_values, y_values, 'bo', linestyle="--")
        plt.imshow(self.image)

    def plot_image_skeleton_keypoints(self, score_coordinate):
        for score, from_c, to_c in score_coordinate:
            #print((score, from_c, to_c))
            if score is not None:
                from_c = [from_c[0], to_c[0]]
                to_c = [from_c[1], to_c[1]]
                plt.plot(from_c, to_c, 'bo', linestyle="--")
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

    def show(self, index=0):
      #  print(len(self.results))
        self.results[index].show(self.skeleton)
        
    def get_paf_map(self, sigma, optimized=False):
        #print(self.results[0].shape)
        img_shape = (640, 480)#, 3) #self.results[0].shape
        parity_fields = ParityFields()
        parity_x = torch.zeros(img_shape + (2, ))
        p_1 = torch.tensor((355, 367)).float()
        p_2 = torch.tensor((423, 314)).float()

        img = torch.zeros((img_shape))
        img[p_1[0].long(), p_1[1].long()] = 255
        img[p_2[0].long(), p_2[1].long()] = 255

        if optimized:
            i, j = torch.meshgrid(
                torch.arange(img_shape[0]), 
                torch.arange(img_shape[1]), 
                indexing='ij'
            )
            grid_tensor = torch.dstack([i, j]).float()
            res = parity_fields.function(grid_tensor, p_1, p_2, sigma)
            return res
        else:
            for i in range(300, img_shape[0]):
                for j in range(300, img_shape[1]):
                    res = parity_fields.unoptimized_function(torch.tensor([i, j]), p_1, p_2, 5)
                    if torch.is_tensor(res):
                        img[i, j] = 128
                        parity_x[i, j] = res
            return img

    def get_metadata(self, index):
        return {
            "skeleton": self.skeleton,
            "keypoints": self.results[index].keypoints,
            "path":  get_local_dir("train2017/" + self.results[index].name)
        }

if __name__  == "__main__":
    obj = Coco()
    obj.load_annotations()
    # obj.show(4)
#    obj.show(6)
  
    
#    print(obj.skeleton)
#    print(obj.results[0].keypoints)
#    print(obj.results[0].name)
