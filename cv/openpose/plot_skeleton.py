from coco import Coco
import numpy as np
from scipy.optimize import linear_sum_assignment

#dataset = Coco().load_annotations().show()
#print(dataset)



"""
We generate a paf field + confidence map

-> We want to output the keypoints from that 

-> Confidence map only used for non-maxium suppression of the detections ? 

-> Okay, so we are mostly working with with the LC for the skeletion
"""
import torch
from parity_fields import ParityFields

img_shape = (640, 480)
parity_fields = ParityFields()

# Known keypoints
p_1 = torch.tensor((355, 367)).float()
p_2 = torch.tensor((423, 314)).float()

def f(u, d_1, d_2):
    p = (
        1 - u 
    ) * d_1 + u * d_2
    res = parity_fields.unoptimized_function(p, p_1, p_2, 5)
    if not torch.is_tensor(res):
        res = torch.tensor([0, 0]).float()
    res = res @ (p_2 - p_1) / torch.norm(p_2 - p_1)
    return res

def trapezoidal(d_1, d_2, n=3):
    dx = 1 / n
    # (1 - u) * dj + udj
    results = 0
    for i in range(0, n):
        print(i)
        x_k = dx * i
        if i > 0 and (i + 1) < n:
            results += 2 * f(x_k, d_1, d_2)
        else:
            results +=  f(x_k, d_1, d_2)
    return dx/2 * results


def E(d_1, d_2):
    print(d_1 , d_2)
    return trapezoidal(d_1, d_2)


if __name__ == "__main__":
    # okay, so the results of E is the matrix we want to maximize
    # Okay, so this is the confidence of the association.
    # <- Experiment:
    # Create a simple matrix 2d of points 
    # <- P_0 = Weight of results at that point. 
    # <- P_1 = Weight of ??
    # E = Total weight (I think)
    # We supply points to E.
    results_1 = E(p_1, p_2)
    print(results_1)
    """
    Hm, okay, I guess.
    """
    results_2 = E(p_1 / 2, p_2)
    print(results_2)
    """
    Okay, so now you know the weights of a given point!

    Excellent :)

    p_1 -> p_2 weight 0.667
    p_3 -> p_2 weight 0
    """
    G = np.array([
                # Nodes 1 -> Nodes 2
                [results_1, results_2],
        ])
    row_indices, col_indices = linear_sum_assignment(G, maximize=True)
    row_names = ['P_2', 'P_3']
    col_names = ['P_1']
    edges = [((row_names[r], col_names[c]), G[r, c])
             for r, c in zip(row_indices, col_indices)]
    print(edges)


