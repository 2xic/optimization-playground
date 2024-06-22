### [LONGNET: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/pdf/2307.02486.pdf)
Use dilated attention to be scale up.
- Drop in replacement fro existing attention layer
- Much less compute intensive (Figure 5)

[tweet](https://twitter.com/giffmana/status/1676864336764055552?s=12)

### [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf)
- Transformer
  - Inputs
    - (reward if index > 0)
    - Latent representation of the observation
    - Action
  - Output
    - Deterministic hidden state
  - Final output
    - Reward
    - Gamma
    - Next latent state
- The observation comes from an autoencoder.
  - This is based on DreamerV2
- Transformer-XL
- Actor and critic policy is trained on top of this imaginary trajectory.

### [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf)
[Source code](https://github.com/eloialonso/iris)

- This is the same idea as in [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf)

### [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039)
- This is the same general idea as in [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf)
  - They use BEAM search for the planning though

### [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202v1.pdf)
- [Gated linear units](https://paperswithcode.com/method/glu)
-  Seemed to show promising results on the attached benchmarks
-  

## [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf)
[Website](https://sites.google.com/berkeley.edu/decision-transformer)
- Input: Prev state + reward + (Action)
- Output: Next action
- Framing it all as sequence modelling problem$$ -> skipping one of the pitfalls with RL (credit assignment problem)
- [Pseudo code](https://arxiv.org/pdf/2106.01345.pdf#page=5) is super simple -> All you need is predict the action given (s, r, a, t)

### [Vision Transformers with Mixed-Resolution Tokenization](https://arxiv.org/pdf/2304.00287.pdf)
- Instead of doing word tokens, do image patches

### [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/pdf/2103.05247.pdf)
Heard about it first [here](https://www.youtube.com/watch?v=Elxn8rS88bI).

The TLDR: they trained a language model transfers on a language task, then transferred learned it on non language downstream tasks. This model would beat some model trained on only the downstream task.

### [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748.pdf)
- Core of it "We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches" (direct quote from the abstract)
  - DIT blocks are similar to the transformers blocks
  - Noise latent + timestep is the input
  - Images look nice

### [ITRANSFORMER: INVERTED TRANSFORMERS ARE EFFECTIVE FOR TIME SERIES FORECASTING](https://arxiv.org/pdf/2310.06625)
[Hackernews thread](https://news.ycombinator.com/item?id=37848321)

[Source code](https://github.com/thuml/iTransformer)

- Based on the tests then run this looks quite promising
- The entire time-series is a token (wtf?). 
- [Reddit](https://old.reddit.com/r/MachineLearning/comments/175ep6x/r_tsinghua_university_inverting_transformers/) thread mentions [TSMixer](https://arxiv.org/pdf/2303.06053)

### [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/pdf/2404.02258)
- Use a router to chose among potential computation paths
- THe options are 
  - Standard block computation
  - Residual connection

### [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/pdf/2404.02905)
[Website](https://var.vision/)

[Github](https://github.com/FoundationVision/VAR)

[Youtube Video](https://www.youtube.com/watch?v=yJ396Ksiv2s) which has some great coverage on this.

- Instead of predicting the next image patch token, they try to predict the next resolution token.
- General flow
  - Have one image that is then created into multiple resolutions
  - Send them through VqVAE to get some representation (tokens)
    - Which should be more efficient and generate fewer tokens that the normal ImageGPT style patches.
  - The goal of the transformer is to predict those representations
- In section 3.1 they have 3 points for why this is better than the existing method. TLDR - structure is lost with the ImageGPT style method, it's inefficient and there is no causal relationship.
- 

