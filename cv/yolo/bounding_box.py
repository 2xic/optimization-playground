from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageBoundingBox:
    def __init__(self) -> None:
        self.bounding_box = [

        ]

    def open(self, image_path, label_path):
        self.image = Image.open(image_path)
        with open(label_path, "r") as file:
            for i in file.read().split("\n"):
                class_id, x_center, y_center, width, height = i.split(" ")
                image_x, image_y = self.image.size
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
                self.bounding_box.append((
                    image_x_bottom, image_y_bottom, image_x_top, image_y_top))
        return self

    def show(self):            
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        for (x_b, y_b, x_t, y_t) in self.bounding_box:
            rect = patches.Rectangle((x_b, y_b), x_t - x_b, y_t - y_b, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()

    def _get_coordinate(self, image_size, center, size):
        center = float(center)
        size = float(size)
        return (
            image_size * center - image_size * size,
            image_size * center + image_size * size,
        )

if __name__ == "__main__":
    ImageBoundingBox().open(
        "test/feature.jpg",
        "test/labels.txt"
    ).show()
