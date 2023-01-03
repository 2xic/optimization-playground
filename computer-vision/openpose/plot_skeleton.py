from coco import Coco
import numpy as np
from scipy.optimize import linear_sum_assignment
from skeleton import Skeleton
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


if __name__ == "__main__":
    # okay, so the results of E is the matrix we want to maximize
    # Okay, so this is the confidence of the association.
    # <- Experiment:
    # Create a simple matrix 2d of points 
    # <- P_0 = Weight of results at that point. 
    # <- P_1 = Weight of ??
    # E = Total weight (I think)
    # We supply points to E.
    obj = Skeleton(
        img_shape=(640, 480),
        skeleton=[[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
        keypoints=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (325, 160, 2), (398, 177, 2), (0, 0, 0), (437, 238, 2), (0, 0, 0), (477, 270, 2), (287, 255, 1), (339, 267, 2), (0, 0, 0), (423, 314, 2), (0, 0, 0), (355, 367, 2)]
    )

    results_1 = obj.E(p_1, p_2, p_1, p_2)
    print(results_1)
    """
    Hm, okay, I guess.
    """
    results_2 = obj.E(p_1, p_2, p_1 / 2, p_2)
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


