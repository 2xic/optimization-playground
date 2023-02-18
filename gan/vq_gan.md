VAE = Variational auto encoder

# [VQ-GAN, explained](https://medium.com/geekculture/vq-gan-explained-4827599b7cf2) ([Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2012.09841.pdf))
VQ-VAE started from [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/pdf/1711.00937.pdf), which introduced the concept of a "discrete learnable coodbook" which is a list of vectors with corresponding indexes. This is used to make the output of the encoder discrete (rather than continuos that is done with VAE). 

[The codebook](https://arxiv.org/pdf/1711.00937.pdf) is just an embedding layer that is learned. So the model will be given an input `x`, and pass it through the encoder. Resulting in a latent variable that is then compared with nearest neighbor with the embedding space, and this is then passed to the decoder.

The idea of VQ-GAN is to be able to leverage transformer / self-attention in the most compute efficient way one has to use the idea of a codebook. Instead of using pixels as input tokens, the encoder extracts an encoding that is then matched with the closest vector in the codeblock, and the decoder reconstruct from this vector. In other words, by using the codebook you gain sparsity that helps the transformer.

One thing to note is that the network is trained in two steps, first the network learns the codeblock, and then the network is freezed. Then to train the model to generate one first pass in the codeblock vector and have an transformer learn to predict the correct sequence of tokens. Then those tokens are passed to the decoder. 

I will note that for instance OpenAi with the paper [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) has also shown that it's possible with just the standard GPT like models to learn a image representation, have it predict the next pixel in an image. However, the quality of these images are not on par with VQ-Gan.


# [wandb.ai - VQ-GAN + CLIP (DAL-e mini)](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained--Vmlldzo4NjIxODA)
Image is encoded with VQ-GAN, and caption with BART. Then the BART decoder takes the encoding of both the image and the caption and returns the next token.
In this architecture CLIP is only used to select the best image.

