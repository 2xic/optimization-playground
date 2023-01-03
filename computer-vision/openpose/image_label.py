from typing import List
from helpers import get_local_dir
from keypoint import KeyPoint
import matplotlib.pyplot as plt
from PIL import Image

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
