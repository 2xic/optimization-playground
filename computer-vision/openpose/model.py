import torch.nn as nn
import torch
from skeleton import Skeleton
from loss import loss
from torchvision import transforms
from PIL import Image
from coco import Coco
import torch

"""
TODO: Should do refined stages
"""

def get_conv(output_shape):
    return nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1, dilation=1, stride=1, bias=False)

class CnnBlock(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.same_shape = nn.Conv2d(input_shape, output_shape, kernel_size=1, padding=0)

        self.conv1 = get_conv(output_shape)
        self.conv2 = get_conv(output_shape)
        self.conv3 = get_conv(output_shape)

    def forward(self, x):
        x = self.same_shape(x)
        x_1 = torch.nn.functional.elu(self.conv1(x))
        x_2 = torch.nn.functional.elu(self.conv2(x_1))
        x_3 = torch.nn.functional.elu(self.conv3(x_2 + x))

        return x_3


class Shared(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.block_1 = get_conv(input_shape)
        self.block_2 = get_conv(input_shape)
        self.block_3 = get_conv(input_shape)

    def forward(self, x):
        x_1 = torch.nn.functional.relu(self.block_1(x))
        x_2 = torch.nn.functional.relu(self.block_2(x_1))
        x_3 = torch.nn.functional.relu(self.block_3(x_2))

        return x_3

class ConfidenceStage(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape, 480, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(480, output_shape, kernel_size=1, padding=0)

    def forward(self, x):
        x_4 = torch.relu(self.conv1(x))
        x_5 = (self.conv2(x_4))

        return x_5


class PafStage(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape, 480, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(480, output_shape * 2, kernel_size=1, padding=0)
        self.output_shape = output_shape

    def forward(self, x):
        x_4 = torch.relu(self.conv1(x))
        x_5 = (self.conv2(x_4))

        return x_5


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', weights='VGG11_Weights.IMAGENET1K_V1')
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:10])

        for params in self.vgg.parameters():
            params.requires_grad = False

        self.shared = Shared(input_shape=256)#, output_shape=3)
        self.confidence_stage = ConfidenceStage(input_shape=256, output_shape=17)
        self.paf_stage = PafStage(input_shape=256, output_shape=19)

    def forward(self, x):
        features = self.vgg(x)

        shared = (self.shared(features))
        self.confidence = (self.confidence_stage(shared))
        self.paf = (self.paf_stage(shared))

        return self.confidence, self.paf

    def train_step(self, image, confidence, paf, get_annotation):
        model_confidence, model_paf = self.forward(image)

        return loss(
            predicted_confidence=model_confidence,
            predicted_paf=model_paf,
            get_annotation=get_annotation,
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

    def single_step(self, image, confidence, paf, get_annotation):
        self.optimizer.zero_grad()
        loss = self.model.train_step(image, confidence, paf, get_annotation)
        loss.backward()
        self.optimizer.step()
        return loss

    def step(self, image, confidence, paf, get_annotation, epochs=1_00):
        for i in range(epochs):
            loss = self.single_step(image, confidence, paf, get_annotation)
            if i % 10 == 0:
                print(loss)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = Net().to(device)

    coco = Coco().load_annotations()

    label = coco.get_metadata(6)
    skeleton = Skeleton(
        img_shape=(480, 640),
        skeleton=label['skeleton'],
        keypoints=label['keypoints'],
        bbox=label['bbox']
    )
    skeleton.sigma = 5
    image = transforms.ToTensor()(Image.open(label['path'])).to(device)
    trainer = Trainer(net)
    trainer.step(
        image.reshape((1, ) + image.shape), 
        skeleton.confidence_map().to(device),
        skeleton.paf_field().to(device),
        get_annotation=lambda x: skeleton.annotation_map(x).to(device),
        epochs=10_000 if torch.cuda.is_available() else 1
    )

    torch.save({
        'model': net.state_dict()
    }, 'model.pt')
