
"""
Bounding box score
"""
from typing import List
from constants import Constants
import torch

class GridBox:
    def __init__(self, bounding_boxes, p) -> None:
        self.bounding_boxes = bounding_boxes
        self.p = p

    def raw(self):
        # TODO: Fix this bounding box last item problem
        return torch.cat((self.bounding_boxes[-1], self.p), 0)

def get_coordinates_of_tensor(tensor, constants: Constants, confidence_threshold=0) -> List[GridBox]:
    """
    For each grid -> S x S
    There will be B bounding box predictions, and C class predictions.

    (x, y, w, h) is used for a bounding box prediction.

    """
    # grid
    collected_bounding_boxes = []
    grid = tensor.reshape((constants.GRID_SIZE, constants.GRID_SIZE,
                          (5 * constants.BOUNDING_BOX_COUNT + constants.CLASSES)))
    for i in range(constants.GRID_SIZE):
        for j in range(constants.GRID_SIZE):
            first_grid_cell = grid[i][j]

            bounding_boxes = [
                # x, y, w, h, confidence
                first_grid_cell[i*5:i*5+5] for i in range(0, constants.BOUNDING_BOX_COUNT)
            ]
            assert len(bounding_boxes) == constants.BOUNDING_BOX_COUNT

            class_score = first_grid_cell[5 * constants.BOUNDING_BOX_COUNT:]

            assert class_score.shape[-1] == constants.CLASSES

            bounding_boxes = [
                i for i in bounding_boxes if i[-1] > confidence_threshold
            ]
            if len(bounding_boxes) > 0:
                collected_bounding_boxes.append(GridBox(
                    bounding_boxes,
                    class_score
                ))
    """
    TODO:
        Add the IOU threshold
        Add the confidence threshold.

    TODO + :
        Add non-maximal suppression 
            - in the paper they say it's 2-3 improved results
    """

    return collected_bounding_boxes
