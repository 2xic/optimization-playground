from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor

def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1
    return out
        
def ones(i: int) -> TT["i"]:
#    return torch.ones(i).long()
    # or not sure if zero is allowed
    return (torch.tensor(range(i)) >= 0).long() 

test_ones = make_test("one", ones, ones_spec, add_sizes=["i"])
run_test(test_ones)
