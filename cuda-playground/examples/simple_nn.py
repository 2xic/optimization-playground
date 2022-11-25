# TODO: cuda playground version of https://iamtrask.github.io/2015/07/12/basic-python-network/

import cudaplayground as p

# TODO: Fix both directional support -> 
syn0 = p.tensor((3,4)).rand() * 2 - 1
syn1 = p.tensor((4,1)).rand() * 2 - 1

X = p.pare_array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
X.print()

y = p.pare_array([[0,1,1,0]]).T()
y.print()

l1 = X.matmul(syn0)
l1.print()

l2 = l1.matmul(syn1)
l2.print()

# l2_delta = (y - l2)*(l2*(1-l2))
# TODO
# - exp func
# - + / * / + for both ways arguments
# 