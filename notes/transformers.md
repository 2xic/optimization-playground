### [LONGNET: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/pdf/2307.02486.pdf)
Use dilated attention to be scale up.
- Drop in replacement fro existing attention layer
- Much less compute intensive (Figure 5)

[tweet](https://twitter.com/giffmana/status/1676864336764055552?s=12)

### [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf)
- TODO ? 

### [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf)
- TODO ? 

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

