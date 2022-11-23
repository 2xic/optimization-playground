# TODO: cuda playground version of https://iamtrask.github.io/2015/07/12/basic-python-network/

import cudaplayground as p

# TODO: Fix both directional support -> 
syn0 = p.tensor((3,4)).rand() * 2 - 1
syn1 = p.tensor((4,1)).rand() * 2 - 1
