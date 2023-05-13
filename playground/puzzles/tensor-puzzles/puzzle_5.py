from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor


def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1


def eye(j: int) -> TT["j", "j"]:
    """
    I assume this is not okay

    a = torch.zeros((j, j))
    a[torch.arange(j),torch.arange(j)] += 1
    """
    
    # I think scatter_ is okay ? 
    return torch.zeros(
        (j, j)
    ).scatter_(1, torch.arange(j).unsqueeze(1), 1).long()


test_eye = make_test("eye", eye, eye_spec, add_sizes=["j"])

run_test(test_eye)
