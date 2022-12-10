import torch.nn as nn
import torch
import torch.nn.functional as F
"""
Model is based on DIspnet
-> https://arxiv.org/pdf/1512.02134.pdf

"""


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=7,
            stride=2,
            dilation=1,
            padding=3,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.conv_3a = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.conv_3b = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_4a = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_4b = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_5a = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_5b = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_6a = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_6b = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        ## pr6 + loss6
        self.pr_6 = nn.Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        """
        
        =================================================0
        
        """

        self.up_conv_5 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.iconv5 = nn.Conv2d(
            in_channels=1025,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pr_5 = nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.up_conv_4 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256    ,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.iconv4 = nn.Conv2d(
            in_channels=769,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pr_4 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )



        self.up_conv_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128    ,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.iconv3 = nn.Conv2d(
            in_channels=385,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pr_3 = nn.Conv2d(
            in_channels=128,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )


        self.up_conv_2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64    ,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.iconv2 = nn.Conv2d(
            in_channels=193,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pr_2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )


        self.up_conv_1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32    ,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.iconv1 = nn.Conv2d(
            in_channels=97,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pr_1 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = x
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(conv1)
        x = self.conv_3a(conv2)
        conv3b = self.conv_3b(x)
        x = self.conv_4a(conv3b)
        conv4b = self.conv_4b(x)
        x = self.conv_5a(conv4b)
        conv5b = self.conv_5b(x)
        x = self.conv_6a(conv5b)
        x = self.conv_6b(x)

        pr_6 = self.pr_6(x)

        assert pr_6.shape[-2] == 12
        assert pr_6.shape[-1] == 6
        """
        First half of the model ^ 
        """

        x = self.up_conv_5(x)

        print(x.shape)
        print(conv5b.shape)
        print(pr_6.shape)


        """
        todo: pr_6 should also be here, but not sure based on the shape
        """
        iconv_input = torch.concat(
            (
                x,
                conv5b,
          #      torch.zeros((1, 1, 24, 12))
         #       pr_6
                F.interpolate(pr_6, scale_factor=2, mode='bilinear', align_corners=False)
            ),
            dim=1
        )
        icon5 = self.iconv5(iconv_input)
        pr_loss_5 = self.pr_5(icon5)


        x = self.up_conv_4(icon5)
        iconv4 = self.iconv4(torch.concat(
            (
                x,
                conv4b,
                #torch.zeros((1, 1, 48, 24))
                F.interpolate(pr_loss_5, scale_factor=2, mode='bilinear', align_corners=False)
            ),
            dim=1
        ))
        pr_loss_4 = self.pr_4(iconv4)

        """
        -----
        """
        x = self.up_conv_3(iconv4)
        iconv3 = self.iconv3(torch.concat(
            (
                x,
                conv3b,
                # pr_loss_4
               # torch.zeros((1, 1, 96, 48))
                F.interpolate(pr_loss_4, scale_factor=2, mode='bilinear', align_corners=False)
            ),
            dim=1
        ))
        pr_loss_3 = self.pr_3(iconv3)

        """
        -----
        """
        x = self.up_conv_2(iconv3)
        iconv2 = self.iconv2(torch.concat(
            (
                x,
                conv2,
                # pr_loss_3
                #torch.zeros((1, 1, 192, 96))
                F.interpolate(pr_loss_3, scale_factor=2, mode='bilinear', align_corners=False)
            ),
            dim=1
        ))
        pr_loss_2 = self.pr_2(iconv2)
        """
        ----
        """
        x = self.up_conv_1(iconv2)
        print(conv1.shape)
        print(x.shape)
        iconv1 = self.iconv1(torch.concat(
            (
                x,
                conv1,
                # pr_loss_2
                #torch.zeros((1, 1, 384, 192))
                F.interpolate(pr_loss_2, scale_factor=2, mode='bilinear', align_corners=False)
            ),
            dim=1
        ))

        pr_loss_1 = self.pr_1(iconv1)

        scaled_up_pr_loss_1 = F.interpolate(pr_loss_1, scale_factor=2, mode='bilinear', align_corners=False)

        return x, scaled_up_pr_loss_1


if __name__ == "__main__":
    model = Model()
    image = torch.zeros((1, 6, 768, 384))
    (output, _) = model(image)
    print(output.shape)
