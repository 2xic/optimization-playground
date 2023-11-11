import torch
import random

box_size = 256

def append_to_image(x):
    z = torch.zeros((4, box_size, box_size))
    z[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    return z

def apply_random_noise(image_y):
    segmentation = torch.zeros((1, box_size, box_size))
    is_regression = False
    # 50 / 50 bit flips
    if random.randint(0, 4) <= 2:
        width = random.randint(0, box_size)
        height = random.randint(0, box_size)

        x = random.randint(0, image_y.shape[1] - width)
        y = random.randint(0, image_y.shape[2] - height)

        # we apply some noise and segmentation
        segmentation[:, x:x+width, y:y+height] = 1
        image_y[:, x:x+width, y:y+height] = image_y.max()
        is_regression = True
    return image_y, segmentation, is_regression

def generate_diff_image(x):
    y = x.clone()
    (_, width, height) = x.shape
    for c_width in range(0, width, box_size):
        for c_height in range(0, height, box_size):
            image_x = append_to_image(x[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y = append_to_image(y[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y, image_segmentation, is_regression = apply_random_noise(image_y)

            assert image_x.shape == (4, box_size, box_size), image_x.shape
            assert image_segmentation.shape == (1, box_size, box_size), image_segmentation.shape
            assert image_x.shape == image_y.shape
            assert type(is_regression) == bool
            
            yield image_x, image_y, image_segmentation, is_regression

def apply_segmentation_callback(x, y, callback):
    (_, width, height) = x.shape
    for c_width in range(0, width, box_size):
        for c_height in range(0, height, box_size):
            image_x = append_to_image(x[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y = append_to_image(y[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_segmentation, is_regression  = callback(image_x, image_y)

            assert image_x.shape == (4, box_size, box_size), image_x.shape
            assert image_segmentation.shape == (1, box_size, box_size), image_segmentation.shape
            assert image_x.shape == image_y.shape
            assert type(is_regression) == bool
            
            yield image_x, image_y, image_segmentation, is_regression

def compare_images(x, y):
    width, height = get_max_diff(x, y)
    for c_width in range(0, width, box_size):
        for c_height in range(0, height, box_size):
            image_x = append_to_image(x[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y = append_to_image(y[:4, c_width:c_width+box_size, c_height:c_height + box_size])

            yield image_x, image_y, (c_width, c_height)

def get_max_diff(x, y):
    (_, a, b) = x.shape
    (_, x, y) = y.shape

    return max(a, x), max(b, y)
