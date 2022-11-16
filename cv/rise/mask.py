import torch
import torch.nn.functional as F
import random

def binary_masks(n, h, w, p, divide_size):
    dh, dw = (int(h * 1/divide_size), int(w * 1/divide_size))
    binary_mask = torch.rand(size=(n, 1, dh, dw))
    binary_mask[binary_mask > p] = 1
    binary_mask[binary_mask < p] = 0
   # print(42)    

    C_h = h / dh
    C_w = w / dw

    new_h = (dh + 1) * C_h
    new_w = (dw + 1) * C_w

    scaled_masks = F.interpolate(binary_mask, scale_factor=2,  mode='bilinear')
    dh, dw = scaled_masks.shape[2], scaled_masks.shape[3]

    image_masks = torch.ones((n, 1, h, w))
    for i in range(n):
        delta_h = random.randint(0, h - dh)
        delta_w = random.randint(0, w - dw)
        image_masks[i, 0, delta_h:delta_h+dh, delta_w:delta_w+dw] -= scaled_masks[i,0 ]

    return image_masks

def mask_score(model, image, class_id, n=100, p=0.3, divide_size=8):
    masks = binary_masks(n, image.shape[0], image.shape[1],p=p, divide_size=divide_size)
    pixel_score = torch.zeros((image.shape))
    for i in range(masks.shape[0]):
        output = image.reshape((1, ) + image.shape) * masks[i].reshape((1, ) + image.shape)
        model_output = (model(output.reshape((1,1, ) + image.shape)))
        pixel_score[masks[i, 0] > 0.25] += model_output[0][class_id]
    return pixel_score / n

