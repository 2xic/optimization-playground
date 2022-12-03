# TODO: cuda playground version of https://iamtrask.github.io/2015/07/12/basic-python-network/

import cudaplayground as p
import time

# TODO: Fix both directional support -> 
syn0 = p.tensor((3,4)).rand().cuda()
syn1 = p.tensor((4,1)).rand().cuda()

X = p.pare_array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]).cuda()
y = p.pare_array([[0,1,1,0]]).T().cuda()

start = time.time()

for i in range(10_000):
    l1 = 1/(1+(-((X.matmul(syn0)))).exp() )
    l2 = 1/(1+(-((l1.matmul(syn1)))).exp() )

    l2_delta = (y - l2) * (l2 * (1-l2))
    l1_delta = l2_delta.matmul(syn1.T()) * (l1 * (1-l1))

    syn1 += l1.T().matmul(l2_delta)
    syn0 += X.T().matmul(l1_delta)
    if i % 100 == 0:
        print(i)
 #   break

l1 = 1/(1+(-((X.matmul(syn0)))).exp())
l2 = 1/(1+(-((l1.matmul(syn1)))).exp())
l2.host().print()
print(f"time: {time.time() - start}")

"""
#0f66aea1564a631ee17541335f3e12f9b3179493
time: 56.64261269569397

# Made exp a proper kernel function
time: 31.971983194351196

# Made transpose a proper kernel function
time: 20.08816432952881

# Made setElement not be called
time: 1.7329158782958984

"""
