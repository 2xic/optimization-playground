import torch
from score_bounding_box import get_coordinates_of_tensor

if __name__ == "__main__":
    GRID_SIZE = 7
    BOUNDING_BOX_COUNT = 2
    CLASSES = 20

    tensor = torch.zeros((
        GRID_SIZE * GRID_SIZE * (BOUNDING_BOX_COUNT * 5 + CLASSES)
    ))
    
    output = get_coordinates_of_tensor(tensor,
                                        GRID_SIZE,
                                        BOUNDING_BOX_COUNT,
                                        CLASSES
    )
    