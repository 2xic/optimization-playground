from PIL import Image
import numpy as np

# Approximation
def pooling(array, size=3):
    x = np.zeros((array.shape[0] // size, array.shape[1] // size))
    print(x.shape)
    for index_i, i in enumerate(list(range(0, array.shape[0] // size))):
        for index_j, j in enumerate(list(range(0, array.shape[1] // size))):
            start_i = i * size
            end_i = i * size + size

            start_j = j * size
            end_j = j * size + size
            
            output = array[start_i:end_i, start_j:end_j]
            if len(output) == 0:
                print(array.shape)
                print((start_i, end_i))
                print((start_j, end_j))
                print(output)
                exit(0)

            x[index_i][index_j] = np.median(output)
    return x

# quantization
def lower_color(array):
    # no python support in numpy :'(
    # but we can pretend it's possible
    # use lower number of bits to represent the value
    # uses less space ...
    # you win 
    pass
    
image = np.asarray(Image.open("shannon.png").convert("L"))
pooled = pooling(image, size=6)
output = Image.fromarray(np.uint8(pooled))
output.save("compressed.png")

print(len(np.uint8(image).tobytes()))
print(len(np.uint8(pooled).tobytes()))
