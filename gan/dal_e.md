# [OpenAi Blog on DAL-E v1](https://openai.com/blog/dall-e/)
DAL-E is a transformer language model. The model receives both the image and the text prompt, and is therefore able to recreate the image.

The outputs are good.

# [Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092.pdf)
Goal is to train a transformer to be able to predict the image tokens (and text) autoregressively. Using pixel values directly would take a lot of memory (as we also know form VQ-GAN), and the authors solved this by learning an autoencoder that compresses the image so the transformer need to keep track of less context given the smaller image. Then the image tokens and text tokens are concatenated, and the model should learns the distribution.  


# [Aditya Ramesh blogpost on DAL-E v2 ("creator" of DAL-E](http://adityaramesh.com/posts/dalle2/dalle2.html)
DAL-E v2 builds on a new idea called unCLIP, and is based on diffusion and CLIP. 

DAL-E generates an image in two steps, first it generates a "draft", and then this image is filled with realistic values to make it into a proper image.
The first step a model (called "the prior") generates the CLIP image embedding from the given caption. In the second stage, an diffusion model which is called unCLIP, receives both an corrupted version of the image it's meant to reconstruct, and the clean CLIP image embeddings. 

# [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/pdf/2204.06125.pdf)
unClip is described a bit more here, so the idea is the following (based on figure 2)
- You have a image + caption
  - Image is fed into CLIP
  - Caption is fed into CLIP
    - Caption is then fed into a "diffusion prior" to get an image embedding. 
      - This embedding is then used to to condition the diffusion decoder and to get the final image

The method involves having a dataset (`X`, `y`) of `X` images with `y` captions. 
- The prior is `P(Z_i | y)` where `Z_i` is the CLIP image embedding
- The decoder `P(X | Z_i, y)`, here the `y` caption is optional.

----
# [Beating OpenAI CLIP with 100x less data and compute](https://www.unum.cloud/blog/2023-02-20-efficient-multimodality)
Model is also open source [https://github.com/FreddeFrallan/Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP)

Implements two public papers to make pretraining more efficent
[Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment](https://arxiv.org/abs/2208.13628)
[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)




