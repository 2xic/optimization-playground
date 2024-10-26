

### [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
[Source code](https://github.com/facebookresearch/jepa)

[Paper](https://scontent.fosl1-1.fna.fbcdn.net/v/t39.2365-6/427986745_768441298640104_1604906292521363076_n.pdf?_nc_cat=103&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=UdLNygGK4qYAb4tziq3&_nc_ht=scontent.fosl1-1.fna&oh=00_AfBDEs6-Dpaz5UTY77fubARAcFPji0zQic_vWxP1YiI2QA&oe=662C8871)

- First heard about this arcature design from [A Path Towards Autonomous AI](https://www.youtube.com/watch?v=DokLw1tILlw) by Yann LeCun and he also talked about it on the Lex Fridman [podcast](https://youtu.be/5t1vTLU7s40?feature=shared&t=1564)
  - Prediction representation of Y instead of predicting raw Y.
- Builds upon [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- It's actually just I-JEPA made working for video instead of images.
  - 

### [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/pdf/2301.08243.pdf)
[Source code](https://github.com/facebookresearch/ijepa)

[Blog post](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)

- core idea is simple: "from a single context block, predict the representations of various target blocks in the same image."
- There a few variants of this already (visualized in figure 2)
  - Join-embedding which just is a function f(x,y) which will have a high value for if x and y is similar, and low if they are not.
  - Generative Architecture which is a function f(g(x, z), y) Learns to construct y from a compatible signal x using a decoder network that is conditional on a latent variable z 
  - Join-Embedding prediction architecture learns to predict the embeddings of signal y from compatible signal `X` using a predictor network that is possibly conditional on a latent variable Z
- The overall idea is vitalized in image 3 and 4

## [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906)
Stabilize the learning with self-supervised methods by adding regularization which is
1. Maintain the variance between the two embeddings
2. De-correlate the variables

The code is [here](https://arxiv.org/pdf/2105.04906#page=13).
