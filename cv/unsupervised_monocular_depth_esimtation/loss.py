import torch
import numpy as np
import cv2
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
    """
    image = torch.zeros((image_i.shape[0], image_i.shape[1]))
    for i in range(image_i.shape[0] - 1):
        for j in range(image_i.shape[1] - 1):
            image[i, j] = SSIM(
                image_i[i:i+block_size, j:j+block_size],
                image_j[i:i+block_size, j:j+block_size]
            )
    return image


def c_ap(image_i, image_r):
    block_image = SSIM_block(
        image_i,
        image_r
    )
    alpha = 1
    output = torch.zeros(image_i.shape)
    for i in range(image_i.shape[0]):
        for j in range(image_r.shape[1]):
            output[i, j] = alpha * (
                1 - block_image[i][j]
            ) / 2\
                + (
                1 - alpha
            )\
                *\
                torch.norm(
                image_i[i][j] - image_r[i][j]
            )
    return output.mean()


def gradient(image):
    kernel_x = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    kernel_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])
    image = image.numpy().astype(np.uint8)
    print(image.shape)
    if len(image.shape) == 2:
        image = np.concatenate((
            image,
        ), axis=0).astype(np.uint8)

    dx = cv2.filter2D(image, cv2.CV_8U, kernel_x)
    dy = cv2.filter2D(image, cv2.CV_8U, kernel_y)

    return (
        dx,
        dy
    )


def c_ds(image, disparities):
    output = torch.zeros(image.shape)
    dx_image, dy_image = gradient(image)
    dx_disparities, dy_disparities = gradient(disparities)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def norm(x): return x
            output[i][j] = torch.tensor((
                abs(dx_disparities[i][j]) * np.exp(
                    -norm(dx_image[i][j])
                )
            ) + (
                abs(dy_disparities[i][j]) * np.exp(
                    -norm(dy_image[i][j])
                )
            ))
    return output.mean()

def c_lr(image, disparities):
    # d^l (i, j) -> makes sense
    # d^r (i, j) -> ok
    #           + d^l_{i,j}
    #           but it also has ^
    #           and that I don't understand quite atm
    return torch.zeros((1))


def loss(image, image_reconstruction):
    """
    TODO: This should be applied for each of the losses,
    and not just the final loss.

    TODO: This should be applied for both left and right,
            not just one side.
    """
    lambda_ap = 1
    results_c_ap = c_ap(image, image_reconstruction)

    r = 6
    lambda_ds = 1 / r
    results_c_ds = c_ds(image, image_reconstruction)

    lambda_lr = 1
    results_c_lr = c_lr(image, image_reconstruction)

    return lambda_ap * (results_c_ap) +\
           lambda_ds * (results_c_ds) +\
           lambda_lr * (results_c_lr)

if __name__ == "__main__":
    x = torch.ones((100, 100))
    y = x  # torch.rand((100, 100))

    print(SSIM(x, y).shape)
    print(loss(x, y))
