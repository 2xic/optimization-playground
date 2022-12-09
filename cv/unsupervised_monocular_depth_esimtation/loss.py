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