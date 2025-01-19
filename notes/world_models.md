# World models

## [Genie: Generative Interactive Environments](https://arxiv.org/pdf/2402.15391)
- Foundation world model.
- Learns a world model from unlabelled video data.
- [The model setup](https://arxiv.org/pdf/2402.15391#page=4) with the components being a video tokenizer, action model and a dynamic models that produces the new frames. 
- 


## [Genie2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
[Hackernews thread](https://news.ycombinator.com/item?id=42317903)

- Improved support for diverse 3D worlds and the demos on the page looks promising
- It's a diffusion world model. Similar to what was used in the Genie 1 model. Based on this [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) architecture.


## [Oasis: A Universe in a Transformer](https://www.decart.ai/articles/oasis-interactive-ai-video-game-model)
[Site](https://oasis-model.github.io/) and [Etched blogpost](https://www.etched.com/blog-posts/oasis) on the launch.

[Hackernews thread](https://news.ycombinator.com/item?id=42014650)

[Web app to play](https://oasis.decart.ai/welcome)

[Source code](https://github.com/etched-ai/open-oasis)

- Diffusion transformer
- VIT-VAE
- Running on [etchded](https://www.etched.com/blog-posts/oasis), actually, no, it just seemed that way. There is this [article](https://www.technologyreview.com/2024/10/31/1106461/this-ai-generated-minecraft-may-represent-the-future-of-real-time-video-generation/) and saw this on [Twitter](https://x.com/__tinygrad__/status/1854851587773956569).

### [DIFFUSION MODELS ARE REAL-TIME GAME ENGINES](https://arxiv.org/pdf/2408.14837)
- Two phases of training
  - RL-Agent plays the game and the sessions are recorded
  - Diffusion model is trained to produce the next frame conditioned on the previous + action
- Limitations are related to long term memory.

## [Do generative video models learn physical principles from watching videos?](https://arxiv.org/pdf/2501.09038)
- New dataset for testing physics understanding of a model.
