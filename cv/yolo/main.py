from coco2yolo import Coco2Yolo
from constants import Constants
from bounding_box import ImageBoundingBox
from loss import simple_yolo_loss
from model import Yolo
from score_bounding_box import get_coordinates_of_tensor
import torch.optim as optim
import torch
import pickle

if __name__ == "__main__":
    constants = Constants(
        CLASSES=3,
        BOUNDING_BOX_COUNT=1
    )
    dataset = Coco2Yolo(constants).load_annotations()
    first_image = dataset.load("000000558840.jpg")
    model = Yolo(constants)

    lr = 1e-4 #3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(100):
        print(f"Epoch {epoch}")
        for first_image in dataset.iter(samples=5):
            name = first_image["name"]
            tensor_image = first_image["image"]
            output = model(tensor_image.reshape((1, ) + tensor_image.shape))

            optimizer.zero_grad()
            loss = simple_yolo_loss(
                output[0],
                first_image["yolo_bounding_boxes"],
                constants
            )

            loss.backward()
            optimizer.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'constants': pickle.dumps(constants),
        'loss': loss,
    }, "model_state")
    
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
