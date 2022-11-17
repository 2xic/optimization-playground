import torch.nn as nn
import torch
from skeleton import Skeleton
# Same model (almost) as https://arxiv.org/pdf/1611.08050.pdf
from loss import loss
from torchvision import transforms
from helpers import get_local_dir
from PIL import Image
from coco import Coco


class CnnBlock(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)

     #   print((
      #      x_1.shape,
       #     x_2.shape,
        #    x_3.shape,
        # ))

        output = torch.concat(
            [
                x_1,
                x_2,
                x_3
            ], dim=1
        )

        return output


class Stage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = CnnBlock()
        self.block_2 = CnnBlock(n=9)
        self.block_3 = CnnBlock(n=9)
        self.conv1 = nn.Conv2d(9, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, 17, kernel_size=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = self.conv1(x_3)
        x_5 = self.conv2(x_4)

        return x_5


class Stage2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = CnnBlock(n=20)
        self.block_2 = CnnBlock(n=9)
        self.block_3 = CnnBlock(n=9)
        self.conv1 = nn.Conv2d(9, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, 19 * 2, kernel_size=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = self.conv1(x_3)
        x_5 = self.conv2(x_4)

        return x_5.reshape((1, 19, 480, 640, 2))


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = Stage1()
        self.stage_2 = Stage2()

    def forward(self, x):
        self.confidence = self.stage_1(x)
        y = torch.concat(
            [
                x,
                self.confidence
            ],
            dim=1
        )
        self.paf = self.stage_2(y)
        return self.confidence, self.paf

    def train_step(self, x, confidence, paf):
        model_confidence, model_paf = self.forward(x)
       # print(model_confidence.shape)
       # exit(0)
   #     print(model_confidence.shape)
  #      print(model_paf.shape)

 #       print(confidence.shape)
#        print(paf.shape)

        return loss(
            model_confidence,
            model_paf,
            confidence=confidence.reshape((1,) + confidence.shape),
            paf=paf.reshape((1,) + paf.shape)
        )


class Trainer:
    def __init__(self, model: Net) -> None:
        self.optimizer = torch.optim.Adam(
            model.parameters()
        )
        self.model = model

    def step(self, x, confidence, paf):
        for i in range(1_00):
            self.optimizer.zero_grad()
            loss = self.model.train_step(x, confidence, paf)
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(loss)
         #   break

if __name__ == "__main__":
    coco = Coco().load_annotations()

    label = coco.get_metadata(6)
    obj = Skeleton(
        img_shape=(480, 640),
        skeleton=label['skeleton'],
        keypoints=label['keypoints']
    )
    net = Net()
    image = transforms.ToTensor()(Image.open(label['path']))
    trainer = Trainer(net)
    trainer.step(
        image.reshape((1, ) + image.shape), 
        obj.confidence_map(),
        obj.paf_field()
    )

    confidence, paf = net( image.reshape((1, ) + image.shape))
    items = list(obj.merge(
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
