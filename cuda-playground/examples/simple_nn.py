# TODO: cuda playground version of https://iamtrask.github.io/2015/07/12/basic-python-network/

import cudaplayground as p

# TODO: Fix both directional support -> 
syn0 = p.tensor((3,4)).rand()# - 1
syn1 = p.tensor((4,1)).rand()# - 1

X = p.pare_array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = p.pare_array([[0,1,1,0]]).T()

for i in range(10_000):
    l1 = 1/(1+p.exp(-((X.matmul(syn0)))))
    l2 = 1/(1+p.exp(-((l1.matmul(syn1)))))

    l2_delta = (y - l2) * (l2 * (1-l2))
    l1_delta = l2_delta.matmul(syn1.T()) * (l1 * (1-l1))

   # print("=" * 32)

  #  l2_delta.print()
 #   l1_delta.print()
#    break

    syn1 += l1.T().matmul(l2_delta)
    syn0 += X.T().matmul(l1_delta)

    syn0.print()
    syn1.print()

#    break

l1 = 1/(1+p.exp(-((X.matmul(syn0)))))
l2 = 1/(1+p.exp(-((l1.matmul(syn1)))))
l2.print()

# TODO
# - exp func
# - + / * / + for both ways arguments
