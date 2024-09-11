Applying machine learning to solve some issue / challenge.

## OpenAI request for search
[OpenAi requests for research v1](https://web.archive.org/web/20190213165912/https://openai.com/requests-for-research/)

[OpenAi requests for research v2](https://openai.com/research/requests-for-research-2)

Some project they mention that seem cool
- Apply [differential Training Of Rollout Policies](https://www.researchgate.net/publication/2312638_Differential_Training_Of_Rollout_Policies) but with NN
- Create a parallelized version of TRPO (I say any RL algorithm, with the goal being to become good at optimizing the code)
  - They for instance mentions `Parameter Averaging in Distributed RL`
- Learned Data Augmentation

## University of Bergen thesis idea's
[Available Master's thesis topics in machine learning](https://www.uib.no/en/rg/ml/128703/available-masters-thesis-topics-machine-learning)

## Other
- [Image to code](https://huggingface.co/docs/transformers/main/tasks/image_captioning#train)
  - Train model to generate source code for an image
- Revert an embedding (text or image)
  - Similar to [this](https://arxiv.org/pdf/2310.06816.pdf) 
- Sim2Real experiment
- Use RL to optimize some SQL query / Something else
- Training a simple driving simulator (https://arxiv.org/pdf/1608.01230.pdf) with same dataset as the speed challenge 
- Train a GAN / Autoencoder to add invisible watermarks 
  - Similar to [this](https://github.com/ShieldMnt/invisible-watermark#supported-algorithms)
- Writing some kernel fusions.
  - I.e some clean version of [micrograd](https://github.com/karpathy/micrograd) with optimizations and such.
- Puzzles
  - [GPU Puzzles](https://github.com/srush/GPU-Puzzles)
  - [Transformer Puzzles](https://github.com/srush/Transformer-Puzzles)
  - [LLM training puzzle](https://github.com/srush/LLM-Training-Puzzles)
  - [Trition puzzles](https://github.com/srush/Triton-Puzzles)
- Implement [spreading vectors for similarity search](./notes/search.md)
- Implement [Cycle gan](https://arxiv.org/pdf/1703.10593)
- Using ML for generating of fuzzing inputs
- Look at, and ideally break down core parts of the paper and code for [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) [Adding Conditional Control to Text-to-Image Diffusion Models(code)](https://github.com/lllyasviel/ControlNet)
- Take a look at [feature-across-time](https://github.com/EleutherAI/features-across-time/tree/main)
