### [LONGNET: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/pdf/2307.02486.pdf)
Use dilated attention to be scale up.
- Drop in replacement fro existing attention layer
- Much less compute intensive (Figure 5)

[tweet](https://twitter.com/giffmana/status/1676864336764055552?s=12)

### [TRANSFORMER-BASED WORLD MODELS ARE HAPPY WITH 100K INTERACTIONS](https://arxiv.org/pdf/2303.07109.pdf)
- TODO ? 

### [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202v1.pdf)
- [Gated linear units](https://paperswithcode.com/method/glu)
-  Seemed to show promising results on the attached benchmarks
-  

## [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf)

[Website](https://sites.google.com/berkeley.edu/decision-transformer)
- Input: Prev state + reward + (Action)
- Output: Next action
- Framing it all as sequence modelling problem -> skipping one of the pitfalls with RL (credit assignment problem)
- [Pseudo code](https://arxiv.org/pdf/2106.01345.pdf#page=5) is super simple -> All you need is predict the action given (s, r, a, t)
- 

