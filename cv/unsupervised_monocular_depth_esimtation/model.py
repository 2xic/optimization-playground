import torch.nn as nn
import torch
import torch.nn.functional as F
"""
Model is based on DispNet
-> https://arxiv.org/pdf/1512.02134.pdf

"""


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(
            #            in_channels=6,
            in_channels=3,
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
        # pr6 + loss6
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
            out_channels=256,
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
            out_channels=128,
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
            out_channels=64,
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
            out_channels=32,
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
        conv1 = torch.relu(self.conv_1(x))
        conv2 = torch.relu(self.conv_2(conv1))
        x = torch.relu(self.conv_3a(conv2))
        conv3b = torch.relu(self.conv_3b(x))
        x = torch.relu(self.conv_4a(conv3b))
        conv4b = torch.relu(self.conv_4b(x))
        x = torch.relu(self.conv_5a(conv4b))
        conv5b = torch.relu(self.conv_5b(x))
        x = torch.relu(self.conv_6a(conv5b))
        x = torch.relu(self.conv_6b(x))
        pr_6 = torch.relu(self.pr_6(x))

        assert pr_6.shape[-2] == 6
        assert pr_6.shape[-1] == 12
        """
        First half of the model ^
        """

        """
        Stage 5
        """
        x = torch.relu(self.up_conv_5(x))
        icon5 = torch.relu(self.iconv5(torch.concat(
            (
                x,
                conv5b,
                F.interpolate(pr_6, scale_factor=2,
                              mode='bilinear', align_corners=False)
            ),
            dim=1
        )))
        pr_loss_5 = torch.relu(self.pr_5(icon5))
        """
        Stage 4
        """
        x = torch.relu(self.up_conv_4(icon5))
        iconv4 = torch.relu(self.iconv4(torch.concat(
            (
                x,
                conv4b,
                F.interpolate(pr_loss_5, scale_factor=2,
                              mode='bilinear', align_corners=False)
            ),
            dim=1
        )))
        pr_loss_4 = torch.relu(self.pr_4(iconv4))
        """
        Stage 3
        """
        x = torch.relu(self.up_conv_3(iconv4))
        iconv3 = torch.relu(self.iconv3(torch.concat(
                                        (
                                            x,
                                            conv3b,
                                            F.interpolate(pr_loss_4, scale_factor=2,
                                                          mode='bilinear', align_corners=False)
                                        ),
                                        dim=1
                                        )))
        pr_loss_3 = torch.relu(self.pr_3(iconv3))

        """
        Stage 2
        """
        x = torch.relu(self.up_conv_2(iconv3))
        iconv2 = torch.relu(self.iconv2(torch.concat(
            (
                x,
                conv2,
                F.interpolate(pr_loss_3, scale_factor=2,
                              mode='bilinear', align_corners=False)
            ),
            dim=1
        )))
        pr_loss_2 = torch.relu(self.pr_2(iconv2))
        """
        Stage 1
        """
        x = torch.relu(self.up_conv_1(iconv2))
        iconv1 = torch.relu(self.iconv1(torch.concat(
            (
                x,
                conv1,
                F.interpolate(pr_loss_2, scale_factor=2,
                              mode='bilinear', align_corners=False)
            ),
            dim=1
        )))
        pr_loss_1 = torch.sigmoid(self.pr_1(iconv1))
        scaled_up_pr_loss_1 = F.interpolate(
            pr_loss_1, scale_factor=2, mode='bilinear', align_corners=False)

        """
        TODO : one of the modifications done to the network was to output two disparity maps.
                This one currently uses DispNet and does not have that
            
            Solved this by a hack with bilinear 2x
        """

        return pr_loss_1, {
            "d_l": (F.interpolate(scaled_up_pr_loss_1, scale_factor=2, mode='bilinear', align_corners=False)[:, :, :384, :768]),
            "d_r": (F.interpolate(scaled_up_pr_loss_1, scale_factor=2, mode='bilinear', align_corners=False)[:, :, 384:, 768:]),
            "losses": [
                pr_loss_1,
                pr_loss_2,
                pr_loss_3,
                pr_loss_4,
                pr_loss_5,
            ]
        }

    def forward_apply(self, left, right):
        _, metadata = self.forward(left)
        d = metadata["d_r"][0]

        # Is this correct ? Looks like they use matmul
        right_reconstructed = d * left

        _, metadata = self.forward(right)

        d = metadata["d_l"][0]
        left_reconstruction = d * right

        return {
            "left_input": left,
            "left_reconstructed": left_reconstruction,
            "right_input": right,
            "right_reconstructed": right_reconstructed
        }


if __name__ == "__main__":
    model = Model()
    image = torch.zeros((1, 6, 768, 384))
    (output, _) = model(image)
    print(output.shape)
