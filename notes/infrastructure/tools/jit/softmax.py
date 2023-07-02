import triton
import triton.language as tl
# MAYBE ? 
#from numba import cuda as numba_cuda
from timer import TimeIt
import torch
import triton
import triton.language as tl
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE FROM https://openai.com/blog/triton/
@triton.jit
def triton_softmax(Y, stride_ym, stride_yn, X, stride_xm, stride_xn, M, N):
    # row index
    m = tl.program_id(0)
    # col indices
    # this specific kernel only works for matrices that 
    # have less than BLOCK_SIZE columns
    BLOCK_SIZE = 1024
    n = tl.arange(0, BLOCK_SIZE)
    # the memory address of all the elements
    # that we want to load can be computed as follows
    X = X + m * stride_xm + n * stride_xn
    # load input data; pad out-of-bounds elements with 0 
    x = tl.load(X, mask=n < N, other=-float('inf'))
    # compute numerically-stable softmax
    z = x - tl.max(x, axis=0)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    y = num / denom
    # write back to Y
    Y = Y + m * stride_ym + n * stride_yn
    tl.store(Y, y, mask=n < N)

@torch.jit.script
def torch_softmax(tensor):
    max_val = torch.max(tensor, 1, keepdim=True)[0]
    z = tensor - max_val
    top = z.exp()
    bottom = torch.sum(z, dim=1, keepdim=True)
    return top / bottom

if __name__ == "__main__":
    epochs = 1_00
    N = 1_000
    N_range = list(range(100, N))
    torch_time_n = []
    trition_time_n = []
    for N in N_range:
        print(N)
        trition_time = TimeIt('trition')
        for _ in range(epochs):
            with trition_time() as x:
                X = torch.normal(0, 1, size=(100, N), device='cuda')
                Y = torch.empty_like(X)
                # SPMD launch grid
                grid = (X.shape[0], )
                # enqueue GPU kernel
                triton_softmax[grid](Y, Y.stride(0), Y.stride(1), 
                            X, X.stride(0), X.stride(1),
                            X.shape[0]    , X.shape[1])

        torch_time = TimeIt('torch')
        for _ in range(epochs):
            with torch_time() as x:
                X = torch.normal(0, 1, size=(100, N), device='cuda')
                y = torch_softmax(X)  
        #print(torch_time.times)
        torch_time_n.append(
            (sum(torch_time.times) / len(torch_time.times)) / 1e6
        )
        trition_time_n.append(
            (sum(trition_time.times) / len(trition_time.times))/ 1e6
        )
    
    plt.plot(N_range, torch_time_n, label="torch")
    plt.plot(N_range, trition_time_n, label="triton")

    plt.xlabel('Shape (100, X)')
    plt.ylabel('Time (MS)')
    plt.legend(loc='upper left')
    plt.savefig('softmax.png')
