import torch.nn as nn
import torch
from skeleton import Skeleton
# Same model (almost) as https://arxiv.org/pdf/1611.08050.pdf
from loss import loss
from torchvision import transforms
from helpers import get_local_dir
from PIL import Image
from coco import Coco
import torch

"""
TODO: Should do refined stages

- TODO: should try to fix the masks for the model
"""

class CnnBlock(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.same_shape = nn.Conv2d(input_shape, output_shape, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.same_shape(x)
        x_1 = torch.relu(self.conv1(x))
        x_2 = torch.relu(self.conv2(x_1))
        x_3 = torch.relu(self.conv3(x_2 + x))

        return x_3


class Shared(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.block_1 = CnnBlock(input_shape, input_shape)
        self.block_2 = CnnBlock(input_shape, input_shape)
        self.block_3 = CnnBlock(input_shape, input_shape)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)

        return x_3

class Stage1(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, output_shape, kernel_size=1)

    def forward(self, x):
        x_4 = torch.relu(self.conv1(x))
        x_5 = torch.relu(self.conv2(x_4))

        return x_5


class Stage2(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, output_shape * 2, kernel_size=1)
        self.output_shape = output_shape

    def forward(self, x):
        x_4 = torch.relu(self.conv1(x))
        x_5 = torch.relu(self.conv2(x_4))

        return x_5


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:10])

        self.shared = Shared(input_shape=256)#, output_shape=3)
        self.stage_1 = Stage1(input_shape=256, output_shape=17)
        self.stage_2 = Stage2(input_shape=256, output_shape=19)

    def forward(self, x):
        with torch.no_grad():
            features = self.vgg(x)

        shared = (self.shared(features))
        self.confidence = (self.stage_1(shared))
        self.paf = (self.stage_2(shared))

        return self.confidence, self.paf

    def train_step(self, x, confidence, paf, annotation):
        model_confidence, model_paf = self.forward(x)

        return loss(
            predicted_confidence=model_confidence,
            predicted_paf=model_paf,
            annotation=annotation,
            confidence=confidence.reshape((1,) + confidence.shape),
            paf=paf.reshape((1,) + paf.shape)
        )


class Trainer:
    def __init__(self, model: Net) -> None:
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4
        )
        self.model = model

    def step(self, x, confidence, paf, annotation):
        for i in range(1_00):
            self.optimizer.zero_grad()
            loss = self.model.train_step(x, confidence, paf, annotation)
            loss.backward()
            self.optimizer.step()
          #  break

            if i % 10 == 0:
                print(loss)
         #   break

if __name__ == "__main__":
    coco = Coco().load_annotations()

    label = coco.get_metadata(6)
    skeleton = Skeleton(
        img_shape=(480, 640),
        skeleton=label['skeleton'],
        keypoints=label['keypoints']
    )
    skeleton.sigma = 5

    net = Net()
    image = transforms.ToTensor()(Image.open(label['path']))
    trainer = Trainer(net)
    trainer.step(
        image.reshape((1, ) + image.shape), 
        skeleton.confidence_map(),
        skeleton.paf_field(),
        skeleton.annotation_map(),
    )

    confidence, paf = net( image.reshape((1, ) + image.shape))
    items = list(skeleton.merge(
        confidence[0],
        paf[0]
    ))

    coco.results[6].plot_image_skeleton_keypoints(items)


    # output = net.train_step(
    #    torch.zeros((1, 3, 640, 480)),
   #     obj.confidence_map(),
  #      obj.paf_field()
 #   )
#    print(output)
#    print(output.shape)
