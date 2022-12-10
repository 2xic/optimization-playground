import torch
import numpy as np
import cv2
import torch.nn.functional as F

"""
training loss

eq_1
    - \lambda_ap = 1 
        * (
            c^l_ap
            +
            c^r_ap
        )
            -> c_ap = 1/N * (
                1 - SSIM(i, j)
            ) / 2
            + (
                1 - \alpha
            ) * (
                reconstruction
                - 
                original image
            )
            -> SSIM = 3x3 block filter
            -> \alpha = 0.85

    - \lambda_ds = 0.1 / r
        -> r = down scaling factor
            -> 1 / N * (
                sum (
                    disparity gradients x
                ) 
                * 
                exp(
                    -image gradient x
                )
                +
                sum (
                    disparity gradients y
                ) 
                * 
                exp(
                    -image gradient y
                )
            )

    - \lambda_lr = 1
        ->  1/ N *
            sum(
                |
                    disparity_r
                    -
                    disparity_l
                |
            )
    -
"""


def SSIM(i, j):
    L = 255  # 2 ** 8 - 1
    k_1 = 0.01
    k_2 = 0.03

    mu_i = torch.mean(i)
    mu_j = torch.mean(j)

    var_i = torch.var(i)
    var_j = torch.var(j)

    N = i.shape[0] * i.shape[1]
    conv = (
        (i - mu_i) *
        (j - mu_j)
    ).sum()
    conv /= (1 / (N))

    c_1 = (k_1 * L) ** 2
    c_2 = (k_2 * L) ** 2
    c_3 = c_2 / 2

    l_xy = (
        (2 * mu_i * mu_j + c_1) /
        (mu_i ** 2 + mu_j ** 2 + c_1)
    )

    c_xy = (
        (2 * mu_i * mu_j + c_2) /
        (var_i ** 2 + var_j ** 2 + c_2)
    )

    s_xy = (
        conv + c_3
    ) / (
        mu_i * mu_j + c_3
    )

    return (
        l_xy ** 1 *
        c_xy ** 1 *
        s_xy ** 1
    )


def SSIM_block(image_i, image_j, block_size=8):
    """
        This is slow.

        Switched to piqa instead.
    """
    image = torch.zeros((image_i.shape[0], image_i.shape[1]))
    for i in range(image_i.shape[-2] - 1):
        for j in range(image_i.shape[-1] - 1):
            image[i, j] = SSIM(
                image_i[i:i+block_size, j:j+block_size],
                image_j[i:i+block_size, j:j+block_size]
            )
    return image


def c_ap(image_i, image_r):
    from piqa import SSIM
    # TODO fix problem with numbewr scaling
    block_image = SSIM().forward(torch.sigmoid(image_i), torch.sigmoid(image_r))
    alpha = 0.7
    output = torch.zeros(image_i.shape)

    output += (
        1 - block_image
    ) / 2 + (1 - alpha) * torch.norm((image_i - image_r), dim=0)

    return (alpha * output).mean()


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


def c_lr(image, disparities):
    # d^l (i, j) -> makes sense
    # d^r (i, j) -> ok
    #           + d^l_{i,j}
    #           but it also has ^
    #           and that I don't understand quite atm
    return torch.zeros((1))


def loss(
        image_l, image_reconstruction_l,
        image_r, image_reconstruction_r,
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
    results_c_lr = c_lr(image_l, image_reconstruction_l)

    return lambda_ap * (results_c_ap) +\
        lambda_ds * (results_c_ds) +\
        lambda_lr * (results_c_lr)


if __name__ == "__main__":
    x = torch.ones((100, 100))
    y = x  # torch.rand((100, 100))

    print(SSIM(x, y).shape)
    print(loss(x, y))
