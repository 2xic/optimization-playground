import torch

LEARNING_RATE = 0.0002
BATCH_SIZE = 32
Z_SHAPE = 100
# Trying to scale up MNIST data, and see what happens
SCALE = 2
IMG_SHAPE_X, IMG_SHAPE_Y = 28 * SCALE, 28 * SCALE

NORMALIZE_INPUTS_WITH_TANH = True
APPLY_AUGMENTATIONS = False
USE_MAX_LOSS = False

ITERATIONS_OF_DISCRIMINATOR_BATCH = 1
ITERATIONS_OF_GENERATOR_BATCH = 1

DEBUG = False

#GET_NOISE_SAMPLE = lambda batch_size: torch.randn(size=(batch_size, 100)).to('cuda')

def GET_NOISE_SAMPLE(batch_size):
    X = torch.zeros((batch_size, Z_SHAPE))
    for i in range(batch_size):
        X[i] = torch.normal(mean=0.5, std=torch.arange(1., float(Z_SHAPE + 1)))
    return X.to('cuda')
