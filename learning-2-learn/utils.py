import torch
# page 11 in the paper https://arxiv.org/pdf/1606.04474.pdf


def preprocess_gradient(grad: torch.tensor):
    p = 10

    has_above_limit = torch.abs(grad) >= torch.exp(-p)
    #        return torch.log(grad) / p
    coordinate = torch.zeros((2, ) + grad.shape)

    coordinate[0, has_above_limit] = torch.log(grad[has_above_limit]) / p
    coordinate[1, has_above_limit] = torch.sgn(grad[has_above_limit])

    coordinate[0, ~has_above_limit] = -1
    coordinate[1, ~has_above_limit] = torch.exp(p) * grad[~has_above_limit]
