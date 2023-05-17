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
from non_maxiumu_supression import soft_nms, nms
import matplotlib.pyplot as plt
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


if __name__ == "__main__":
    # hot dog
    categories = [58]
    constants = Constants(
        # Currently class loss is not implemented
        CLASSES=len(categories),
        BOUNDING_BOX_COUNT=1
    )
    dataset = Coco2Yolo(
        constants,
        categories=categories
    ).load_annotations(
        samples=10
    )
    model = Yolo(constants).to(device)

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    yolo_dataset = YoloDataset(
        dataset,
        constants=constants,
    )
    dataloader = DataLoader(
        yolo_dataset,
        batch_size=16,
        shuffle=True,
    )
    EPOCHS = 1_000
    loss_over_time = []
    for epoch in range(EPOCHS):
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
                    )
                    for i in range(y.shape[0])
                ]
            )
            # print(loss)
            # print(y.shape, X.shape)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(epoch, total_loss / batches)
        loss_over_time.append((total_loss / batches))

        if epoch % 100 == 0 or (EPOCHS - 1) == epoch:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'constants': pickle.dumps(constants),
                'loss': loss,
            }, "model_state")

            print("lets try out the model output then!")

            for example_idx, file_idx in enumerate([0, 2, 3]):
                results = yolo_dataset.get_raw_index(file_idx)
                if results is None:
                    continue
                name = results["name"]
                tensor_image = results["image"].to(device)
                tensor_image = tensor_image.reshape((1, ) + tensor_image.shape)
                output = None

                with torch.no_grad():
                    output = model(tensor_image)
                    output = get_coordinates_of_tensor(
                        output,
                        constants,
                        confidence_threshold=0.65
                    )

                image = ImageBoundingBox()
                image.load_image(name, constants)

                boxes = []
                scores = []
                for i in output:
                    for j in i.bounding_boxes:
                        cords = ImageBoundingBox().convert_yolo_2_box(
                            constants.image_width, constants.image_height,
                            *j[:4]
                        )
                        if cords is not None:
                            boxes.append(
                                cords
                            )
                            scores.append(round(j[-1].item(), 2))

                for (i, score) in nms(*soft_nms(boxes, scores)):
                    print(i)
                    image.bounding_box.append(i)
                    image.confidence.append(score)
        #            image.load_bbox(i, 1)

                image.save(f'predicted_{example_idx}.png')

                ImageBoundingBox().load_original_image(name).save_with_raw_coco(
                    results["raw_bounding_boxes"],
                    name=f'example_{example_idx}.png'
                )
            plt.plot(loss_over_time)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig('loss.png')
            plt.clf()
