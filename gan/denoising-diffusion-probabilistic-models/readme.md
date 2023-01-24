
# [Paper](https://arxiv.org/pdf/2006.11239.pdf)
Diffusion models are parametrized markov chains, and the model learns the transitions.

The training algorithm is described as 
1. Sample an image from the true distribution
2. Select t as an uniform number between 1 and T (T is the max step taken in the chain)
3. Sample noise from the gaussian with N(0, I) (I believe I here means that it's diagonal)
4. Loss is MSE of the the actual noise and our model predicted noise (Which is defined using a timestep function)

---

### Links

[ What are Diffusion Models? ](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[  The Annotated Diffusion Model ](https://huggingface.co/blog/annotated-diffusion)

