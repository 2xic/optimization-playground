import torch

def get_random_matrix_vector():
    n = 11
    matrix = torch.normal(mean=1, std=0.5, size=(n, n))
    vector = torch.normal(mean=0.5, std=0.5, size=(1, n))
#    print(matrix.shape)
#    print(vector.shape)
    return matrix, vector

def get_quadratic_function_error(matrix, vector):
    # goal is to optimize 
    # f(lambda) = ||W * lambda - y|| ** 2
    return lambda x: torch.norm((matrix  * x - vector)) ** 2


if __name__ == "__main__":
    print(get_quadratic_function_error(
        *get_random_matrix_vector()
    )(torch.arange(11)))
