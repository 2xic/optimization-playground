import torch
import torchvision
import random

file = "./dataset/964c833e5724a19908e2010141a16a54e9e0fd3f78ab5a0dfb4b08075859a540_desktop.png"

x = torchvision.io.read_image(file)
box_size = 256

def append_to_image(x):
    z = torch.zeros((4, box_size, box_size))
    z[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    return z

def apply_random_noise(image_y):
    segmentation = torch.zeros((1, box_size, box_size))
    if random.randint(0, 128) < 32:
        x = random.randint(0, image_y.shape[0])
        y = random.randint(0, image_y.shape[1])
        width = random.randint(0, 32)
        height = random.randint(0, 32)

        # noise + segmentation
        segmentation[:, x:x+width, y:y+height] = 1
        image_y[:, x:x+width, y:y+height] = 255
      #  print("noisit")
    return image_y, segmentation

def generate_diff_image(x):
    y = x.clone()
    (_, width, height) = x.shape
    for c_width in range(0, width, box_size):
        for c_height in range(0, height, box_size):
            image_x = append_to_image(x[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y = append_to_image(y[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y, image_segmentation = apply_random_noise(image_y)

            assert image_x.shape == (4, box_size, box_size), image_x.shape
            assert image_segmentation.shape == (1, box_size, box_size), image_segmentation.shape
            assert image_x.shape == image_y.shape
            yield image_x, image_y, image_segmentation

def compare_images(x, y):
    (_, width, height) = x.shape
    for c_width in range(0, width, box_size):
        for c_height in range(0, height, box_size):
            image_x = append_to_image(x[:4, c_width:c_width+box_size, c_height:c_height + box_size])
            image_y = append_to_image(y[:4, c_width:c_width+box_size, c_height:c_height + box_size])

            yield image_x, image_y, (c_width, c_height)
