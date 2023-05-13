from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor

def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]
        
def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[torch.arange(a.shape[0]), torch.arange(a.shape[0])] 


test_diag = make_test("diag", diag, diag_spec)
run_test(test_diag)
