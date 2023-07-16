# [Image generation with shortest path diffusion](https://arxiv.org/pdf/2306.00501.pdf)

## Paper notes

Diffusion models learns to progressively reverse a corrupted image. There has been some discussion around what the best way to corrupt an image is and this paper looks at how well corruption using a "minimized path taken" (Fisher metric is proposed).

The general equation that will be used thought the paper is this
$$\Sigma_0 = \mathcal{FDF^\dagger}$$
Here $\mathcal{F}$ is the discrete fourier transform and $D$ is the diagonal matrix with power of the spectrum data. Quote from paper "$\dagger$ denotes the operations of matrix transpose and complex conjugation" which is possible to get from torch with `th.fft.fft2(vector, dim=(-2, -1), norm='ortho')`


The shortest path diffusion. Given that we have a Gaussian distribution $X \sim \mathcal{N}({0, \Sigma})$ where `X` is the image and `\Sigma` is the covariance describing the image. Going from the image corruption at time `t=0` to `t=T` will only involve changes to the covariance matrix. The question then is, what is the shortest path ?
This is described in theorem 3.1, but I need to look a bit more on that.

For the image corruption looks to follow a similar form to other diffusion models, but they have created an optimal matrix for applying the noise. 
This is described in theorem 3.2, but here I also need to look at it a bit more.

Applying this all to real image they use some tricks to make this all more compute efficient by doing approximations namely for the $\mathcal{D}$ part of the equation. 

### The algorithm for shortest path diffusion 
DFT = Discrete Fourier Transform

1. Compute the power spectrum of the dataset
2. Compute the optimal filter from t=0 to t=T
3. While not converged
   1. Sample a item `X_0` from the dataset and compute it's DFT `u_0`
   2. Sample uniformly `t`
   3. Sample noise $\epsilon_t$ and compute its noise in frequency space $\epsilon_t$
   4. Compute the corrupted $u_t$ and its inverse DFT `x_t`
   5. One step optimize the $\text{loss}(g_\omega(x_t), \epsilon_t)$

Corruption happens in the frequency space and not the pixel space. 
$$u_t = \mathcal{F}^\dagger X_t$$
Image is recovered like this 
$$x_t = \mathcal{F}^\dagger u_t$$

### The algorithm fro image generation
DFT = Discrete Fourier Transform

1. You have trained the network
2. Set noise $\sigma_t$ for all $t$ to $T$
3. t = T
4. Sample $X_t \sim \mathcal{N}(0, I)$ and compute its DFT $u_T$
5. While t > 0
   1. Sample  $z_t \sim \mathcal{N}(0,I)$
   2. Compute $u_{t - 1}$ and its inverse DFT $x_{t-1}$
   3. $t = t - 1$
6. Return $x$

### Implementation
They have published the source code here https://github.com/mtkresearch/shortest-path-diffusion


