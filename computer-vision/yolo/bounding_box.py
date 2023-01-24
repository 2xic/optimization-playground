import matplotlib.pyplot as plt
import matplotlib.patches as patches
from constants import Constants
from helpers import convert_image

class ImageBoundingBox:
    def __init__(self) -> None:
        self.bounding_box = [

        ]

    def load_image(self, image_path, constants: Constants):
        self.image = convert_image(image_path, constants)
        return self

    def open(self, image_path, label_path):
        self.load_image(image_path)
        with open(label_path, "r") as file:
            for i in file.read().split("\n"):
                class_id, x_center, y_center, width, height = i.split(" ")
                self.load_bbox(
                    [x_center, width, y_center, height]
                )
        return self

    def show(self):
        _, ax = plt.subplots()
        ax.imshow(self.image)
        for (x_b, y_b, x_t, y_t) in self.bounding_box:
            #print((x_b, y_b, x_t, y_t))
            rect = patches.Rectangle(
                (x_b, y_b), x_t - x_b, y_t - y_b, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    def load_bbox(self, yolo_bbox):
        x_center, width, y_center, height = yolo_bbox
        image_x, image_y = self.image.size
        self.convert_yolo_2_coco(
            image_x,
            image_y,
            x_center,
            width,
            y_center,
            height
        )
        return self

    def convert_yolo_2_coco(self, image_x, image_y, x_center, width, y_center, height):
        image_x_bottom, image_x_top = self._get_coordinate(
            image_x,
            x_center,
            width
        )
        image_y_bottom, image_y_top = self._get_coordinate(
            image_y,
            y_center,
            height
        )
        cords = (
            image_x_bottom, image_y_bottom, image_x_top, image_y_top)

        if False not in [self.is_valid_size(x) for x in cords]:
            self.bounding_box.append(cords)
        else:
            pass
        return self

    def is_valid_size(self, x):
        return 0 <= x 

    def _get_coordinate(self, image_size, center, size):
        center = float(center)
        size = float(size)
        return (
            image_size * center - image_size * size,
            image_size * center + image_size * size,
        )


if __name__ == "__main__":
    """
    ImageBoundingBox().open(
        "test/feature.jpg",
        "test/labels.txt"
    ).show()
    """
    ImageBoundingBox().load_image(
        "000000558840.jpg",
        Constants()
    ).load_bbox(
        (0.4772736728191376, 0.07791886478662491, 0.4721715450286865, 0.07105270773172379)
    ).show()