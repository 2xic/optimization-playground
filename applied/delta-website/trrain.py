from model import get_model
from dataset import generate_diff_image, x, compare_images
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision


def check_delta(model, x, y):
    x = torchvision.io.read_image(x)
    y = torchvision.io.read_image(y)
    output = torch.zeros((1, ) + y.shape[1:]) * 255
    
    for (image_x, image_y, location) in compare_images(x, y):
        combined_image = torch.concat((
            image_x,
            image_y
        ), dim=0)
        combined_image = torch.unsqueeze(combined_image, 0)
        with torch.no_grad():
            output_segmentation = model(combined_image)[0]
            (loc_x, loc_y) = location
            (size_x, size_y) = (min(256, x.shape[1] - loc_x), min(256, x.shape[2] - loc_y))

            print(((loc_x,loc_x+size_x), (loc_y,loc_y+size_y)), output.shape)
            output[:, loc_x:loc_x+size_x, loc_y:loc_y+size_y] = output_segmentation[:, :size_x, :size_y]
    return output

model = get_model()
eps = 1e-10
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss = 0
index = 0
for image_x, image_y, image_segmentation in generate_diff_image(x):
    combined_image = torch.concat((
        image_x,
        image_y
    ), dim=0)
    combined_image = torch.unsqueeze(combined_image, 0)
    image_segmentation = torch.unsqueeze(image_segmentation, 0)
    output_segmentation = model(combined_image)
   
    weights = 1
    # adding eps so it dosen't get stuck
    img_loss = nn.BCELoss()(output_segmentation + eps, image_segmentation)
    #print(img_loss)
    if torch.isnan(img_loss):
        raise Exception("Loss is nan")
    loss += img_loss
    print(f"Loss {loss} {index}")

    if index % 4 == 0:
        loss.backward()
        optimizer.step()
        model.zero_grad()
        loss = 0
    index += 1
    break

torchvision.utils.save_image(check_delta(
    model,
    'dataset/0d2bf7a3a2ed2f18ab8b1b20583486bd2d478ec95730bf20242bd84584dee603_mobile.png',
    'dataset/964c833e5724a19908e2010141a16a54e9e0fd3f78ab5a0dfb4b08075859a540_mobile.png'
), 'test.png')
exit(0)



