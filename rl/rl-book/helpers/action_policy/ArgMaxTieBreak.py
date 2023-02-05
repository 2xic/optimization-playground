import numpy as np

def argmax_tie_break(numpy_array):
    return np.random.choice(np.where(numpy_array == numpy_array.max())[0])
