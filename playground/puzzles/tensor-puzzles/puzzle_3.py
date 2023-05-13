from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor

def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]
            
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    """
    -> https://pytorch.org/docs/stable/generated/torch.outer.html
    -> should be like sum gradually
    -> 
    """
    c = ((torch.zeros((a.shape[0], b.shape[0]))) + a.reshape((-1, 1)))\
        * (b.reshape((1, -1)))
    return c.long()
  #  min_shape = min(a.shape[0], b.shape[0])
  #  return a[:min_shape] * b[:min_shape]
    
test_outer = make_test("outer", outer, outer_spec)
run_test(test_outer)
