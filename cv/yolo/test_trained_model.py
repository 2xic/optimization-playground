import torch
from model import Yolo
import pickle
import torch.optim as optim
from coco2yolo import Coco2Yolo
from bounding_box import ImageBoundingBox

checkpoint = torch.load("model_state")
constants = pickle.loads(checkpoint['constants'])
model = Yolo(constants)
lr = 1e-4 #3
optimizer = optim.Adam(model.parameters(), lr=lr)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("Loaded !")

dataset = Coco2Yolo(constants).load_annotations()
name = "000000558840.jpg"
first_image = dataset.load(name)
tensor_image = first_image["image"]
print(tensor_image.shape)

output = model(tensor_image.reshape((1, ) + tensor_image.shape))
predicted_grid = output[0].reshape((constants.GRID_SIZE, constants.GRID_SIZE,
                                    (5 * constants.BOUNDING_BOX_COUNT + constants.CLASSES)))

print(predicted_grid)
image = ImageBoundingBox()
image.load_image(name, constants)
#for i in predicted_grid:
for i in range(constants.GRID_SIZE):
    for j in range(constants.GRID_SIZE):
        image.load_bbox(predicted_grid[i][j][:4])
image.show()
