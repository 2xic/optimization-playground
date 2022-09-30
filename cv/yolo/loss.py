import torch
from constants import Constants

def yolo_loss(predicted, truth, constants: Constants):
    """
        psuedo code based on the paper

        lambda_cord = 5
        lambda_noobj = 0.5
    
        lambda_cord 
            for each grid cell (i) S^2
                for each bounding box (j)
                    if jth bounding box is responsible for bounding box i
                        - hm ?
                        - loss is based on predicted x and y (MSE)
        + 
        lambda_cord
            for each grid cell (i) S^2
                for each bounding box (j)
                    if jth bounding box is responsible for bounding box i ()
                        - loss is based on predicted width and height (MSE)
        + 
            for each grid cell (i) S^2
                for each bounding box (j)
                    if jth bounding box is responsive for bounding box i()
                        - loss is based on the class prediction (mse)
        +
        lambda_nooobj
            for each grid in cell (i) 
                for each bounding box (j)
                    - loss is based on class prediction (mse)
        +
            for each grid in cell (i) 
                if object appears in cell i
                    - for each classes
                        - MSE of probability of class ,and actual probability.
    """


    """
        We assume the output of the tensor being 
    """
    loss = torch.zero(1)
    for i in predicted:
        for b in range(constants.BOUNDING_BOX_COUNT):
            pass            


