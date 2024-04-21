## just a though experiment on whether ml is more driven by the engineering ideas or new algorithms from the research sides

**DISCLAIMER**: Only some thoughts, my view might change tomorrow. 

My general view though is that a lot of ML breakthroughs are the result of [convex tinkering](https://en.wikipedia.org/wiki/Nassim_Nicholas_Taleb)

# Computer Vision
- Most of the ideas here feels like it was done a while back. CNN created in the 90s. What changed was compute and scale. Engineering driven.

## Alexnet
- [Alex Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky) had been working on [cuda kernels](https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/convnet_report.pdf) and optimizing them. That is the reason Alexnet took off. There was no brand new research idea in that sense (CNN already existed). See for instance 
- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997.pdf)

## [Resnet](https://arxiv.org/abs/1512.03385)
- Super simple idea, but works extremely well. "Convex thinking" might be / could be the reason for the discovery.

# Reinforcement learning
- Might require some more of the research side, but given that we have the bellman equation etc. it also might be more a engineering feat.
- Many of the great RL success stories are the result of clever engineering 

## OpenAI Five
- [https://arxiv.org/pdf/1912.06680.pdf](https://arxiv.org/pdf/1912.06680.pdf) the algorithm used is PPO scaled up.

## AlphaZero
- Some cleaver ideas mixed together
  - Self play
  - Use NN for value function and controlling the search
  - Monte carlo search tree 

## ChatGPT
- Scaled up transformers. Engineering driven.

### Transformers
- Attention mechanism used came from research (but I guess this can be debated)
- The model itself I would say comes more from the engineering side.

## Generative models
### GANs
- More on the research side, but I think a cleaver engineer could have thought of something similar. 
- [Ian Goodfellow](https://youtu.be/Z6rxFNMGdn0?feature=shared&t=1616) also credited (some) alcohol with this discovery (lol).

### Diffusion models
- This is more of a research idea. Knowing some physics [helps](https://en.wikipedia.org/wiki/Diffusion_model). That said, the underlying idea of predicting the state transition might as well be engineering driven, but you wouldn't likely been able to get a nicely working model.

## Resources
- [Nassim Taleb on Scientific Discovery](https://www.science.org/content/blog-post/nassim-taleb-scientific-discovery) on the convex tinkering (i.e "theory comes out of practice")
- [Deep Learning ideas that have stood the test of time](https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/) has some good paper with some of the greater ideas
