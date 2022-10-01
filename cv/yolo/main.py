from statistics import mode
from coco2yolo import Coco2Yolo
from constants import Constants
from bounding_box import ImageBoundingBox
from loss import prediction_2_grid, simple_yolo_loss, yolo_loss
from model import Yolo
from score_bounding_box import get_coordinates_of_tensor
import torch.optim as optim

if __name__ == "__main__":
    constants = Constants(
        CLASSES=3,
        BOUNDING_BOX_COUNT=1
    )
    dataset = Coco2Yolo(constants).load_annotations()
    first_image = dataset.load("000000558840.jpg")
    model = Yolo(constants)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(100):
        #for first_image in dataset.iter():
        if True:
            name = first_image["name"]
            tensor_image = first_image["image"]

            print(tensor_image.shape)

            output = model(tensor_image.reshape((1, ) + tensor_image.shape))
            """
            old loss

            truth_grid = (prediction_2_grid(
                first_image["yolo_bounding_boxes"], constants, class_id=[1, ] * 20))

            predicted_grid = (prediction_2_grid(
                [
                    i.raw()
                    for i in get_coordinates_of_tensor(output[0], constants, confidence_threshold=0.5)
                ], constants)
            )
            """
            optimizer.zero_grad()
            loss = simple_yolo_loss(
                output[0],
                first_image["yolo_bounding_boxes"],
                constants
            )
#            loss = yolo_loss(predicted_grid, truth_grid, constants)
            loss.backward()
            optimizer.step()

    print("lets try out the model output then!")
    output = model(tensor_image.reshape((1, ) + tensor_image.shape))
    output = get_coordinates_of_tensor(
        output, constants, confidence_threshold=0.5)

    image = ImageBoundingBox()
    image.load_image(name, constants)
    for i in output:
        for j in i.bounding_boxes:
            print(j)
            image.load_bbox(j[:4])
    image.show()
