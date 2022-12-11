import torch
import numpy as np
import cv2
import torch.nn.functional as F

def c_ap(input_image, reconstructed_image):
    from piqa import SSIM
    # TODO fix problem with number scaling
    ssim = SSIM().forward(input_image, reconstructed_image)
    # mentioned in the paper under Appearance Matching Loss
    alpha = 0.85
    output = alpha * (1 - ssim) / 2 +\
                     (1 - alpha) * torch.norm((input_image - reconstructed_image), dim=0)
    return (output).mean()


def gradient(image):
    kernel_x = torch.tensor([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]).reshape((1, 3, 3))
    kernel_y = torch.tensor([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ]).reshape((1, 3, 3))
    kernel_x = torch.concat((
        kernel_x,
        kernel_x,
        kernel_x
    ),
        dim=0).reshape((1, 3, 3, 3)).float()
    kernel_y = torch.concat((
        kernel_y,
        kernel_y,
        kernel_y
    ),
        dim=0).reshape((1, 3, 3, 3)).float()

    dx = F.conv2d(image, kernel_x, padding=1)
    dy = F.conv2d(image, kernel_y, padding=1)

    return (
        dx,
        dy
    )


def c_ds(image, disparities):
    dx_image, dy_image = gradient(image)
    dx_disparities, dy_disparities = gradient(disparities)
    output = torch.zeros(image.shape)

    assert not torch.any((dx_image == torch.nan))
    assert not torch.any((dy_image == torch.nan))
    assert not torch.any((dx_disparities == torch.nan))
    assert not torch.any((dy_disparities == torch.nan))

    output = (
        dx_disparities * torch.exp(-torch.norm(dx_image, dim=0)) +
        dy_disparities * torch.exp(-torch.norm(dy_image, dim=0))
    )

    return output.mean()


def c_lr(disparities_i, disparities_j):
    # d^l (i, j) -> makes sense
    # d^r (i, j) -> ok
    #           + d^l_{i,j}
    #           but it also has ^
    #           and that I don't understand quite atm
    return (
        disparities_i - disparities_j
    ).mean()

def loss(
        image_l, image_reconstruction_l, image_l_d,
        image_r, image_reconstruction_r, image_r_d,
    ):
    # (batch, channel, width, height)
    assert len(image_l.shape) == 4
    assert len(image_reconstruction_l.shape) == 4
    """
    TODO: This should be applied for each of the losses,
    and not just the final loss.
    """
    lambda_ap = 1
    results_c_ap = c_ap(image_l, image_reconstruction_l) +\
                    c_ap(image_r, image_reconstruction_r)

    r = 6
    lambda_ds = 1 / r
    results_c_ds = c_ds(image_l, image_reconstruction_l)+\
                    c_ds(image_r, image_reconstruction_r)
    lambda_lr = 1
    results_c_lr = c_lr(image_l_d, image_r_d) +\
                    c_lr(image_r_d, image_l_d)

    return lambda_ap * (results_c_ap) +\
        lambda_ds * (results_c_ds) +\
        lambda_lr * (results_c_lr)


if __name__ == "__main__":
    x = torch.ones((100, 100))
    y = x  # torch.rand((100, 100))

    print(SSIM(x, y).shape)
    print(loss(x, y))
