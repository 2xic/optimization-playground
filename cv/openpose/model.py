import torch.nn as nn
import torch

# Same model (almost) as https://arxiv.org/pdf/1611.08050.pdf

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

        print((
            x_1.shape,
            x_2.shape,
            x_3.shape,
        ))

        output = torch.concat(
            [
                x_1,
                x_2,
                x_3
            ],dim=1
        )

        return output

class Stage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = CnnBlock()
        self.block_2 = CnnBlock(n=9)
        self.block_3 = CnnBlock(n=9)
        self.conv1 = nn.Conv2d(9, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=1)

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
        self.block_1 = CnnBlock(n=10)
        self.block_2 = CnnBlock(n=9)
        self.block_3 = CnnBlock(n=9)
        self.conv1 = nn.Conv2d(9, 7, kernel_size=1)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = self.conv1(x_3)
        x_5 = self.conv2(x_4)
    
        return x_5

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = Stage1()
        self.stage_2 = Stage2()

    def forward(self, x):
        self.confidence = self.stage_1(x)
        print(x.shape)
        print(self.confidence.shape)
        y = torch.concat(
            [
                x, 
                self.confidence
            ],
            dim=1
        )
        self.paf = self.stage_2(y)

        return self.paf

if __name__ == "__main__":
    net = Net()
    # pretrained feature map input
    output = net(torch.zeros((1, 3, 100, 100)))
    print(output.shape)
