import torch
#from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.dataloaders.CelebA import get_dataloader

EPOCHS = 1_00
LEARNING_RATE = 0.0002
BATCH_SIZE = 32
Z_SHAPE = 100
# Trying to scale up MNIST data, and see what happens
SCALE = 2
IMG_SHAPE_X, IMG_SHAPE_Y = 28 * SCALE, 28 * SCALE

NORMALIZE_INPUTS_WITH_TANH = True
APPLY_AUGMENTATIONS = False
USE_MAX_LOSS = False

"""
Visualzize
"""
PLOT_LOSS_AND_DISCRIMINATOR = False
PLOT_EVERY_EPOCH = False
PLOT_FINAL_OUTPUT_NAME = f"{IMG_SHAPE_X}x{IMG_SHAPE_Y}_{Z_SHAPE}.png"

ITERATIONS_OF_DISCRIMINATOR_BATCH = 1
ITERATIONS_OF_GENERATOR_BATCH = 1

DATALOADER = lambda: get_dataloader(
    batch_size=BATCH_SIZE,
    #overfit=False
) 

DEBUG = False

#GET_NOISE_SAMPLE = lambda batch_size: torch.randn(size=(batch_size, 100)).to('cuda')

def GET_NOISE_SAMPLE(batch_size):
    X = torch.zeros((batch_size, Z_SHAPE))
    for i in range(batch_size):
        X[i] = torch.normal(mean=0.5, std=torch.arange(1., float(Z_SHAPE + 1)))
    return X.to('cuda')
