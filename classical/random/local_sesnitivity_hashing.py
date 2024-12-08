"""
https://zilliz.com/learn/Local-Sensitivity-Hashing-A-Comprehensive-Guide
https://dzone.com/articles/minhash-lsh-implementation-walkthrough
https://softwaredoug.com/blog/2023/08/21/implementing-random-projections
https://myscale.com/blog/implementing-locality-sensitive-hashing-in-python/
"""

import numpy as np 
import torch
from numpy.linalg import norm
from levenstein_distance import levenshtein_distance

def hash_function(datapoint, random_vector):
  projection = np.dot(datapoint, random_vector)
  return 1 if int(projection) > 0  else 0

def generate_random_matrix(num_projections, data_dim):
  return np.random.randn(num_projections, data_dim)

def lsh_hash(datapoint, random_matrix):
  hash_values = []
  for random_vector in random_matrix:
    hash_values.append(hash_function(datapoint, random_vector))
  return torch.tensor(hash_values).reshape((1, -1)).float()

def is_equal(hash_1, hash_2):
 #   print(hash_1)
#    print(hash_2)
#    return torch.nn.functional.cosine_similarity(hash_1, hash_2)
    return torch.cdist(hash_1, hash_2)

def get_hash(datapoint):
    num_projections = 16
    data_dim = len(datapoint)
    random_matrix = generate_random_matrix(num_projections, data_dim)
    hash_values = lsh_hash(datapoint, random_matrix)
    return hash_values

def hamming_distance(string1: str, string2: str) -> int:
    # https://en.wikipedia.org/wiki/Hamming_distance#History_and_applications
    if len(string1) != len(string2):
        raise ValueError("Strings must be of equal length.")
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter / len(string1)

def make_into_str(value: np.array):
   return "".join(list(map(lambda x: str(int(x)), value.tolist()[0])))

datapoint = np.arange(1, 10).reshape((-1))
print(is_equal(get_hash(datapoint), get_hash(datapoint)))
datapoint_2 = np.arange(10, 1, -1).reshape((-1))
# bad threshold 
print(is_equal(get_hash(datapoint_2), get_hash(datapoint_2)))
print(is_equal(get_hash(datapoint), get_hash(datapoint_2)))
# probably won't work 
print(levenshtein_distance(make_into_str(get_hash(datapoint)), make_into_str(get_hash(datapoint_2))))
print(levenshtein_distance(make_into_str(get_hash(datapoint)), make_into_str(get_hash(datapoint))))

# probably won't work 
print(hamming_distance(make_into_str(get_hash(datapoint)), make_into_str(get_hash(datapoint_2))))
print(hamming_distance(make_into_str(get_hash(datapoint)), make_into_str(get_hash(datapoint))))
