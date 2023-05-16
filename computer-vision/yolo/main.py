from coco2yolo import Coco2Yolo
from constants import Constants
from bounding_box import ImageBoundingBox
from loss import simple_yolo_loss
from model import Yolo
from score_bounding_box import get_coordinates_of_tensor
import torch.optim as optim
import torch
import pickle
from dataloader import YoloDataset
from torch.utils.data import DataLoader

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


if __name__ == "__main__":
    categories = [50]
    constants = Constants(
        # Currently class loss is not implemented
        CLASSES=len(categories),
        BOUNDING_BOX_COUNT=1
    )
    dataset = Coco2Yolo(constants, categories=categories).load_annotations()
    model = Yolo(constants).to(device)

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    yolo_dataset = YoloDataset(
        dataset,
        constants=constants,
    )
    dataloader = DataLoader(
        yolo_dataset,
        batch_size=8,
        shuffle=True,
    )

    for epoch in range(20):
        total_loss = 0
        batches = 0
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            optimizer.zero_grad()
            loss = torch.concat(
                [
                    simple_yolo_loss(
                        output[i],
                        y[i],
                        constants,
                        #name=results["name"][i]
                    )
                    for i in range(y.shape[0])
                ]
            ).sum() / X.shape[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(epoch, total_loss / batches)
        """
        for images in dataset.iter(samples=30):
            name = images["name"]
            tensor_image = images["image"].to(device)
            output = model(tensor_image.reshape(
                (1, ) + tensor_image.shape)
            )
            loss = simple_yolo_loss(
                output[0],
                torch.tensor(images["yolo_bounding_boxes"], device=device),
                constants,
            )

            loss.backward()
            total_loss += loss.item()
        print(f"Loss : {total_loss}")
        optimizer.step()
        optimizer.zero_grad()
        """

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'constants': pickle.dumps(constants),
        'loss': loss,
    }, "model_state")

    print("lets try out the model output then!")

    for example_idx, file_idx in enumerate([0, 1, 3]):
        results = yolo_dataset.get_raw_index(file_idx)
        name = results["name"]
        tensor_image = results["image"].to(device)
        output = model(tensor_image.reshape((1, ) + tensor_image.shape))
        output = get_coordinates_of_tensor(
            output,
            constants,
            confidence_threshold=0.7
        )

        image = ImageBoundingBox()
        image.load_image(name, constants)
        for i in output:
            for j in i.bounding_boxes:
                print(j)
                image.load_bbox(j[:4], round(j[-1].item(), 2))
        image.save(f'predicted_{example_idx}.png')

        ImageBoundingBox().load_image(
            name,
            constants
        ).load_bbox(
            results["yolo_bounding_boxes"][0], 1
        ).save(f'example_{example_idx}.png')
