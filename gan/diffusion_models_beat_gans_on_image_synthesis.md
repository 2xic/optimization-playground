# Diffusion Models Beat GANs on Image Synthesis

# [Paper](https://arxiv.org/pdf/2105.05233.pdf)
Paper introduces an improved architecture, and it able to be more sample efficient than BigGan-deep.

Diffusion models work by reversing a gradual noise process. Starting with noise `x_T` and iteratively creates less an less noise samples (`X_(t - 1)`, `X_(t-2)`, ... `X_0`). In the paper the noise is drawn from diagonal Gaussian distribution.
The model should learn to predict the noise component of a noise sample `X_t`. The loss of the model is absolute error between noise and the predicted noise.

-> Stopped at section 2.1, should try to build a simple model before continuing


