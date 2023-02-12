# https://openreview.net/pdf?id=1AxQpKmiTc
# https://arxiv.org/abs/2110.12661
import torch
import math
from scipy.linalg import hadamard
import torch.nn as nn

def set_model_weights(model):
#    for name, param in model.named_parameters():
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, nn.Linear):
            module.weight.data = ZerO_Init_on_matrix(module.weight.data)
            nn.init.constant_(module.bias, 0)
        elif  isinstance(module, nn.Conv2d):
            n = module.weight.data.shape[-1] // 2
            module.weight.data[:, :, n, n] = ZerO_Init_on_matrix(module.weight.data[:, :, n, n])

# Based from the author https://github.com/jiaweizzhao/ZerO-initialization/blob/main/example_mnist.ipynb
def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper.
    
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    
    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    
    return init_matrix.to(matrix_tensor.device)
