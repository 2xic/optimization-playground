from cmath import pi
from inspect import classify_class_attrs
from os import GRND_RANDOM
from typing import List
import torch
from constants import Constants
from coco2yolo import Coco2Yolo
import math


class GridEntry:
    def __init__(self, i, p) -> None:
        self.x = i[0]
        self.y = i[1]
        self.w = i[2]
        self.h = i[3]
        self.confidence = i[4]
        self.class_p = p
        self.p_class_id = torch.tensor(self.class_p).argmax()

class GridEntryTruth(GridEntry):
    def __init__(self, i, class_id) -> None:
        self.x = i[0]
        self.y = i[1]
        self.w = i[2]
        self.h = i[3]
        self.class_id = class_id


def prediction_2_grid(yolo_bounding_boxes, constants: Constants, class_id=None) -> List[GridEntry]:
    """
    Quote from paper:

    Our system divides the input image into an S × S grid.
    If the center of an object falls into a grid cell, that grid cell
    is responsible for detecting that object.
    """
    direction_grid_box_size = 1 / (constants.GRID_SIZE)
    grid_array = [
        [
            [] for _ in range(constants.GRID_SIZE)
        ]
        for _ in range(constants.GRID_SIZE)
    ]
    assert len(grid_array) == constants.GRID_SIZE
    assert len(grid_array[0]) == constants.GRID_SIZE
    assert len(grid_array[0][0]) == 0

    for index, i in enumerate(yolo_bounding_boxes):
        assert len(i) != 5 or len(i) != constants.CLASSES + \
            5, "wrong size of predictions"
        # x, y, w, height, width, (confidence if not truth)
        x, y, _, _, = i[:4]

        assert x <= 1 and x >= 0, f"Expected yolo format for x {x}"
        assert y <= 1 and y >= 0, f"Expected yolo format for y {y}"

        grid_box_x = int(x // direction_grid_box_size)
        grid_box_y = int(y // direction_grid_box_size)

        assert grid_box_x < constants.GRID_SIZE, f"Bad gridbox_x {x} - {direction_grid_box_size} - {grid_box_x} vs {constants.GRID_SIZE}"
        assert grid_box_y < constants.GRID_SIZE, f"Bad gridbox_y {y} - {direction_grid_box_size} - {grid_box_y} vs {constants.GRID_SIZE}"

        if class_id is None:
            grid_array[grid_box_y][grid_box_x].append(
                GridEntry(i, p=i[5:])
            )
        else:
            grid_array[grid_box_y][grid_box_x].append(
                GridEntryTruth(i, class_id=class_id[index])
            )
    return grid_array


def yolo_loss(predicted: List[GridEntry], truth: List[GridEntryTruth], constants: Constants):
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

        ----
        okay I think I get it now (but it's late, so it might be wrong)
        - you weight correct predictions based on first part of the loss.
        - you weight incorrect prediction based on the second loss.
    """

    coordinate_loss = torch.zeros(1)

    """
    Quote from the paper

    Note that the loss function only penalizes classification
    error if an object is present in that grid cell (hence the conditional 
    class probability discussed earlier). It also only penalizes bounding box coordinate 
    error if that predictor is “responsible” for the ground truth box (i.e. has the highest
    IOU of any predictor in that grid cell).
    
    - 
    """

    # if object appears
    for i in range(constants.GRID_SIZE):
        for j in range(constants.GRID_SIZE):
            predicted_grid = predicted[i][j]
            truth_grid = truth[i][j]

            if len(predicted_grid) > 0 and len(truth_grid) > 0:
                predicted_first_boundary = predicted_grid[0]
                truth_boundary = truth_grid[0]

                coordinate_loss += (
                    predicted_first_boundary.x -
                    truth_boundary.x
                ) ** 2 \
                    + (
                    predicted_first_boundary.y -
                        truth_boundary.y
                ) ** 2

                coordinate_loss += (
                    math.sqrt(predicted_first_boundary.w) -
                    math.sqrt(truth_boundary.w)
                ) ** 2 \
                    + (
                    math.sqrt(predicted_first_boundary.h) -
                    math.sqrt(truth_boundary.h)
                ) ** 2
            """
            elif len(predicted_grid) > 0:
                for predicted_first_boundary in predicted_grid:
                    coordinate_loss += (
                        predicted_first_boundary.x
                    ) ** 2 \
                        + (
                        predicted_first_boundary.y
                    ) ** 2

                    coordinate_loss += (
                        math.sqrt(predicted_first_boundary.w)
                    ) ** 2 \
                        + (
                        math.sqrt(predicted_first_boundary.h)
                    ) ** 2
            """
    # if object does not appear.
    no_obj_loss = torch.zeros(1)

    for i in range(constants.GRID_SIZE):
        for j in range(constants.GRID_SIZE):
            predicted_grid = predicted[i][j]
            truth_grid = truth[i][j]

            if len(predicted_grid) > 0:
                truth_class_id = truth_grid[0].class_id if len(
                    truth_grid) > 0 else None

                for first_prediction_grid in predicted_grid:
                    if first_prediction_grid is not None:
                        for (index, p_i) in enumerate(first_prediction_grid.class_p):
                            if index == truth_class_id:
                                no_obj_loss += (1 - p_i) ** 2
                            else:
                                no_obj_loss += (p_i) ** 2

                    if truth_class_id is not None:
                        no_obj_loss += (first_prediction_grid.p_class_id -
                                        truth_class_id) ** 2

    lambda_cord = 5
    lambda_no_obj = 0.5
    print((
        "loss",
        coordinate_loss,
        no_obj_loss
    ))
    loss = lambda_cord * coordinate_loss + lambda_no_obj * no_obj_loss

    return loss
