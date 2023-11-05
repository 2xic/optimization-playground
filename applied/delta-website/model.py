"""
I want a linear model that compress the image

This should then be the input into the 
"""
import torch

def get_model():
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet',
        in_channels=8, 
        out_channels=1, 
    )
    return model
