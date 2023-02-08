import numpy as np

def argmax_tie_break(numpy_array, non_max=None):
    if non_max is not None:
        for index in range(len(numpy_array)):
            if index not in non_max:
                numpy_array[index] = float('-inf')

    return np.random.choice(np.where(numpy_array == numpy_array.max())[0])
