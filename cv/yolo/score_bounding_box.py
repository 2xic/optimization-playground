
"""
Bounding box score
"""


from stat import FILE_ATTRIBUTE_NO_SCRUB_DATA


def get_coordinates_of_tensor(tensor,
                              S, B, C):
    """
    For each grid -> S x S
    There will be B bounding box predictions, and C class predictions.

    (x, y, w, h) is used for a bounding box prediction.

    """
    # grid
    grid = tensor.reshape((S, S, (5 * B + C)))
    first_grid_cell = grid[0][0]
    bounding_boxes =  [
        first_grid_cell[i*5:i*5+5] for i in range(0, B)
    ]
    assert len(bounding_boxes) == B

    class_score = first_grid_cell[5 * B:]

    assert class_score.shape[-1] == C

    """
    TODO:
        Add the IOU threshold
        Add the confidence threshold.

    TODO + :
        Add non-maximal suppression 
    """

    return []
