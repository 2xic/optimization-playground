"""
Context I was reading this article
https://labs.spotify.com/2014/02/28/how-to-shuffle-songs/

Additional recourses 
https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering

"""
from PIL import Image
import numpy as np
from random import randint


def add(image, x, y, value):
    if(x < image.shape[1]):
        if(y < image.shape[0]):
            image[y, x] += value

def get_floyd_steinberg_dithering_image(image):
    image = np.asarray(image).copy()
    for i in range(image.shape[0] ):
        for j in range(image.shape[1] ):
            old = image[i, j]  
            color = 0 if old < 125 else 255 
            image[i, j] = color

            quant_error = (old - color)
            
            add(image, j + 1, i, (7 / 16) * quant_error)
            add(image, j - 1, i + 1, (3 / 16) * quant_error)
            add(image, j, i + 1, (5 / 16) * quant_error)
            add(image, j + 1, i + 1, (1 / 16) * quant_error)
    return image

def get_random_dithering_image(image):
    image = np.asarray(image).copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = 255 if(randint(0, 255) < image[i, j]) else 0
            image[i, j] = color
    return image

if __name__ == "__main__":
    image = Image.open("michelangelodavid.png")
    floyd_steinberg_dithering = Image.fromarray(get_floyd_steinberg_dithering_image(image))
    random_dithering = Image.fromarray(get_random_dithering_image(image))
    floyd_steinberg_dithering.save('floyd_steinberg_dithering.png')
    random_dithering.save('random_dithering.png')


"""
img = Image.open("michelangelodavid.png")
ORG = np.asarray(img).copy()

I = np.asarray(img).copy()

from random import randint
use_lazy = False
'''
lazy, bad version!
'''

if(use_lazy):
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            color = 1 if(randint(0, 255) < I[i, j]) else 0
            I[i, j] = color


def add(I, x, y, value):
    if(x < I.shape[1]):
        if(y < I.shape[0]):
            I[y, x] += value

if not use_lazy:
    for i in range(I.shape[0] ):
        for j in range(I.shape[1] ):
            old = I[i, j]  
            color = 0 if old < 125 else 255 
            I[i, j] = color

            quant_error = (old - color)
            
            add(I, j + 1, i, (7 / 16) * quant_error)
            add(I, j - 1, i + 1, (3 / 16) * quant_error)
            add(I, j, i + 1, (5 / 16) * quant_error)
            add(I, j + 1, i + 1, (1 / 16) * quant_error)


img = Image.fromarray(I)
plt.imshow(img)
plt.show()
"""