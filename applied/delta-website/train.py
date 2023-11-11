from model import get_model
from dataset import compare_images, get_max_diff, box_size
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from dataset_entries import DatasetEntries

debug_cpu = False
device = torch.device('cuda') if torch.cuda.is_available() and not debug_cpu else torch.device('cpu')

def check_delta(model, x, y):
    x = torchvision.io.read_image(x)
    y = torchvision.io.read_image(y)
    (width, height) = get_max_diff(x, y)
    output = torch.ones((1, ) + (width, height)) 
    
    for (image_x, image_y, location) in compare_images(x, y):
        combined_image = torch.concat((
            image_x,
            image_y
        ), dim=0)
        combined_image = torch.unsqueeze(combined_image, 0)
        with torch.no_grad():
            output_segmentation = model(combined_image.to(device))[0]
            (loc_x, loc_y) = location
            (size_x, size_y) = (min(box_size, width - loc_x), min(box_size, height - loc_y))

            output[:, loc_x:loc_x+size_x, loc_y:loc_y+size_y] = output_segmentation[:, :size_x, :size_y] > 0.5
    return output

if __name__ == "__main__":
    model = get_model().to(device)
    eps = 1e-10
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = DatasetEntries().generate_dynamic_diffs(
        "./dataset/964c833e5724a19908e2010141a16a54e9e0fd3f78ab5a0dfb4b08075859a540_desktop.png"
    ).add_same_image_with_no_delta(
        "./dataset/same_version_diff_data/1"
    ).add_same_image_with_with_delta(
        "./dataset/examples_with_diffs"
    )

    for epochs in range(50):
        loss = 0
        index = 0
        print("iterates ? ")
        for results in dataset.iterates():
           # print(len(results))
            for (image_x, image_y, image_segmentation, _is_regression) in results:
                combined_image = torch.concat((
                    image_x,
                    image_y
                ), dim=0)
                combined_image = torch.unsqueeze(combined_image, 0).to(device)
                image_segmentation = torch.unsqueeze(image_segmentation, 0).to(device)
                output_segmentation = model(combined_image)
            
                weights = 1
                # adding eps so it doesn't start outputting nans
                img_loss = nn.BCELoss()(output_segmentation + eps, image_segmentation)
                print(img_loss)
                if torch.isnan(img_loss):
                    raise Exception("Loss is nan")
                loss += img_loss
                print(f"Loss {loss} {index} (epochs=={epochs})")

                if index % 4 == 0:
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    loss = 0
                index += 1

        torchvision.utils.save_image(check_delta(
            model,
            'dataset/0d2bf7a3a2ed2f18ab8b1b20583486bd2d478ec95730bf20242bd84584dee603_mobile.png',
            'dataset/964c833e5724a19908e2010141a16a54e9e0fd3f78ab5a0dfb4b08075859a540_mobile.png'
        ), f'debug/test_{epochs}.png')
