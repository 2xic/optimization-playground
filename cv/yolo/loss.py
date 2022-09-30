from cmath import pi
from os import GRND_RANDOM
import torch
from constants import Constants
from coco2yolo import Coco2Yolo
import math


def prediction_2_grid(yolo_bounding_boxses, constants: Constants):
    """
Q   Quote from paper :

    Our system divides the input image into an S × S grid.
    If the center of an object falls into a grid cell, that grid cell
    is responsible for detecting that object.
    """
    direction_grid_box_size = 1 / (constants.GRID_SIZE)
    grid_array = [
        [
            [] for i in range(constants.GRID_SIZE)
        ]
        for i in range(constants.GRID_SIZE)
    ]
    assert len(grid_array) == constants.GRID_SIZE
    assert len(grid_array[0]) == constants.GRID_SIZE
    assert len(grid_array[0][0]) == 0

    for i in yolo_bounding_boxses:
        assert len(i) != 5 or len(i) != constants.CLASSES + 5, "wrong size of predictions"
        _, x, y, _, _ = i[:5]

        grid_box_x = int(x // direction_grid_box_size)
        grid_box_y = int(y // direction_grid_box_size)

        grid_array[grid_box_y][grid_box_x].append(
            i
        )
    return grid_array


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

        ----
        okay I think I get it now (but it's late, so it might be wrong)
        - you weight correct predictions based on first part of the loss.
        - you weight incorrect prediction based on the second loss.
    """

    loss = torch.zeros(1)
    # if object appears
    for i in range(constants.GRID_SIZE):
        for j in range(constants.GRID_SIZE):
            predicted_grid = predicted[i][j]
            truth_grid = truth[i][j]

            if len(predicted_grid) > 0 and len(truth_grid) > 0:
                predicted_first_boundary = predicted_grid[0]
                if len(predicted_first_boundary) < (5 + constants.CLASSES):
                    raise Exception("Wrong prediction array")

                predicted_truth_boundary = truth_grid[0]

                # TODO: actually this should not depend on the class prediction
                if predicted_first_boundary[0] == predicted_truth_boundary[0]:
                    loss += (
                        predicted_first_boundary[1] -
                        predicted_truth_boundary[1]
                    ) ** 2 \
                        + (
                        predicted_first_boundary[2] -
                            predicted_truth_boundary[2]
                    ) ** 2

                    loss += (
                        math.sqrt(predicted_first_boundary[3]) -
                        math.sqrt(predicted_truth_boundary[3])
                    ) ** 2 \
                        + (
                        math.sqrt(predicted_first_boundary[4]) -
                            math.sqrt(predicted_truth_boundary[4])
                    ) ** 2
    # if object does not appear.
    for i in range(constants.GRID_SIZE):
        for j in range(constants.GRID_SIZE):
            if len(predicted_first_boundary) < (5 + constants.CLASSES):
                raise Exception("Wrong prediction array")

            predicted_grid = predicted[i][j]
            truth_grid = truth[i][j]

            if len(predicted_grid) > 0 and len(truth_grid) > 0:
                truth_class_id = truth_grid[0][0]
                first_prediction_grid = predicted_grid[0]

                for (index, p_i) in enumerate(first_prediction_grid[5:]):
                    if index == truth_class_id:
                        loss += (1 - p_i) ** 2
                    else:
                        loss += (p_i) ** 2                    
    return loss


def test_1():
    coco2Yolo = Coco2Yolo()
    constants = Constants()
    p = [0, ] * constants.CLASSES
    p[4] = 1
    grid_box = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                bounding_boxes=[199.84, 200.46, 77.71, 70.88]
            ))
            + p
        ],
        constants
    )
    grid_box_truth = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                bounding_boxes=[199.84, 200.46, 77.71, 70.88]
            ))

        ],
        constants
    )

    assert yolo_loss(
        grid_box,
        grid_box_truth,
        constants
    ) == 0


def test_2():
    coco2Yolo = Coco2Yolo()
    constants = Constants()

    p = [0, ] * constants.CLASSES
    p[4] = 1

    grid_box = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                # SMALL ADJUSTMENT TO BOUNDING BOX
                bounding_boxes=[198.84, 18.46, 77.71, 70.88]
            ))
            + p
        ],
        constants
    )
    grid_box_truth = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                bounding_boxes=[199.84, 200.46, 77.71, 70.88]
            ))

        ],
        constants
    )

    assert yolo_loss(
        grid_box,
        grid_box_truth,
        constants
    ) != 0

def test_3():
    coco2Yolo = Coco2Yolo()
    constants = Constants()

    p = [0, ] * constants.CLASSES
    p[5] = 1

    grid_box = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                # SMALL ADJUSTMENT TO BOUNDING BOX
                bounding_boxes=[198.84, 18.46, 77.71, 70.88]
            ))
            + p
        ],
        constants
    )
    grid_box_truth = prediction_2_grid(
        [
            [4, ] +
            list(coco2Yolo.coco2yolo(
                width=640,
                height=247,
                bounding_boxes=[199.84, 200.46, 77.71, 70.88]
            ))

        ],
        constants
    )

    assert yolo_loss(
        grid_box,
        grid_box_truth,
        constants
    ) != 0


if __name__ == "__main__":
    test_1()
    test_2()
