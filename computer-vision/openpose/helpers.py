from ast import Constant
import os
from PIL import Image


def get_local_dir(path):
    return os.path.join(
        os.path.dirname(__file__),
        path
    )
