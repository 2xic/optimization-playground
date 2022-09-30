from statistics import mode
from coco2yolo import Coco2Yolo
from constants import Constants
from loss import prediction_2_grid, yolo_loss
from model import Yolo
from score_bounding_box import get_coordinates_of_tensor
import torch.optim as optim

if __name__ == "__main__":
    dataset = Coco2Yolo().load_annotations()
    constants = Constants()
    first_image = dataset.load("000000558840.jpg")
    # print(first_image)
    tensor_image = first_image["image"]
    model = Yolo(constants)

    optimizer = optim.Adam(model.parameters())

    output = model(tensor_image.reshape((1, ) + tensor_image.shape))
    truth_grid = (prediction_2_grid(first_image["yolo_bounding_boxes"], constants))
    predicted_grid = (prediction_2_grid(
        [
            i.raw()
            for i in get_coordinates_of_tensor(output[0], constants)
        ], constants)
    )
    
    optimizer.zero_grad()
    loss = yolo_loss(predicted_grid, truth_grid, constants)
    loss.backward()
    optimizer.step()
    
    print(loss)
