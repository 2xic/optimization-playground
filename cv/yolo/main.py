from statistics import mode
from coco2yolo import Coco2Yolo
from constants import Constants
from bounding_box import ImageBoundingBox
from loss import prediction_2_grid, yolo_loss
from model import Yolo
from score_bounding_box import get_coordinates_of_tensor
import torch.optim as optim

if __name__ == "__main__":
    dataset = Coco2Yolo().load_annotations()
    constants = Constants()
    first_image = dataset.load("000000558840.jpg")
    # print(first_image)
    path = first_image["path"]
    tensor_image = first_image["image"]
    model = Yolo(constants)

    optimizer = optim.Adam(model.parameters())

    for _ in range(10):
        output = model(tensor_image.reshape((1, ) + tensor_image.shape))
        truth_grid = (prediction_2_grid(
            first_image["yolo_bounding_boxes"], constants, class_id=[1]))

        predicted_grid = (prediction_2_grid(
            [
                i.raw()
                for i in get_coordinates_of_tensor(output[0], constants, confidence_threshold=0.5)
            ], constants)
        )

        optimizer.zero_grad()
        loss = yolo_loss(predicted_grid, truth_grid, constants)
        loss.backward()
        optimizer.step()

##        if loss.item() <= 1:
#            break

    print("lets try out the model output then!")
    output = model(tensor_image.reshape((1, ) + tensor_image.shape))
    output = get_coordinates_of_tensor(output, constants)

    image = ImageBoundingBox()
    image.load_image(path)
    for i in output:
        for j in i.bounding_boxes:
            image.load_bbox(j[:4])
    image.show()
