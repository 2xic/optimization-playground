from collections import defaultdict
import json
import glob
from torchvision import transforms
import os 
from PIL import Image

def get_local_dir(path):
    return os.path.join(
        os.path.dirname(__file__),
        path
    )

def get_original_image(image_name):
    image = Image.open(
        get_local_dir(image_name)
    )
    return image

class CocoImageLoader:
    def __init__(self) -> None:
        self.files = glob.glob("./test2017/*.jpg")

    def iter(self):
        for i in self.files:
            yield self.load(i)

    def get_list(self):
        return list(sorted(self.image_bbox.keys()))

    def load(self, image_name):
        image = get_original_image(image_name)
        width, height = image.size
        image = transforms.ToTensor()(
            image
        )
        return {
            "name": image_name,
            "original_size": [width, height],
            "image": image
        }


if __name__ == "__main__":
    example = CocoImageLoader()
    print(json.dumps(example.id_category))
