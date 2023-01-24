from ast import Constant
import os
from PIL import Image


def get_local_dir(path):
    return os.path.join(
        os.path.dirname(__file__),
        path
    )

def convert_image(image_name, constants: Constant):
    image = Image.open(
        get_local_dir("train2017/" + image_name)
    )
    result = Image.new(
        image.mode, (constants.image_width, constants.image_height), (0, 0, 0))
    result.paste(image, (0, 0))
    return result