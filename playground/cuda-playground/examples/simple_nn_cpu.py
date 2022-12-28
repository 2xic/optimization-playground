# TODO: cuda playground version of https://iamtrask.github.io/2015/07/12/basic-python-network/

import cudaplayground as p
import time

start = time.time()
# TODO: Fix both directional support -> 
syn0 = p.tensor((3,4)).rand()
syn1 = p.tensor((4,1)).rand()

X = p.pare_array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = p.pare_array([[0,1,1,0]]).T()

for i in range(10_000):
    l1 = 1/(1+(-((X.matmul(syn0)))).exp())
    l2 = 1/(1+(-((l1.matmul(syn1)))).exp())

    l2_delta = (y - l2) * (l2 * (1-l2))
    l1_delta = l2_delta.matmul(syn1.T()) * (l1 * (1-l1))

    syn1 += l1.T().matmul(l2_delta)
    syn0 += X.T().matmul(l1_delta)

l1 = 1/(1+(-((X.matmul(syn0)))).exp())
l2 = 1/(1+(-((l1.matmul(syn1)))).exp())
l2.print()

print(f"time: {time.time() - start}")
