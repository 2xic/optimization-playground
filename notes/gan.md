## GAN 
For the paper notes that are so short that they don't need their own file.

# [Scaling up GANs for Text-to-Image Synthesis](https://arxiv.org/pdf/2303.05511.pdf)
GigaGan is an attempt to level up GANs so they can "combat" against the new hot diffusion models, they tried scaling up StyleGan, but that did not work, so they ended up creating a new architecture. 
*The results looks quite nice.*
They have a few ideas that I have not seen before related to GANs, one is the usage of CLIP as a pretrained model for the text encoding. They also use a bank of convolution filters to not have to scale up the width of the convolution layers.

# [Cones: Concept Neurons in Diffusion Models for Customized Generation](https://arxiv.org/pdf/2303.05125.pdf)
This paper finds that certain neurons in a diffusion model maps to certain subjects (called concept neurons in the paper), they can be identified by the statistics of network gradients for a given subject.
The paper tries to explore ways to control the concept neurons to generate a subject in diverse settings.
They also propose a algorithm for finding concepts neurons, so if by scaling down a neuron increases it's loss it would be regarded as a concept neuron.
They also show it's possible to combine multiple concepts into a single image, and the results look quite good.
 
# [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175.pdf)
They discuss how the model is able to learn robust, and on robust features and that non robust features are the reason for why adversarial examples are so effective. Then they go into how one is able to detect robust and non robust features, and how to deal with them.
Usually the features are highly predictive for the models, but not us humans, models usually don't look at ears like humans do, but instead something else and more abstract.
They also do some interesting relabeling that seems wrong to humans, but actually improves the model accuracy (figure 1 b)


# [Multi-Adversarial Variational Autoencoder Networks](https://arxiv.org/pdf/1906.06430.pdf)
MAVEN attempts to be a more robust GAN architecture. They use 3 networks, encoder, generator and a discriminator.
The job of the encoder is to compress the input to a lower dimension, the job of the generator is to generate inputs to trick the discriminator, and the discriminators job is to learn what is real and fake.
The main difference between this architecture from a traditional GAN / VAE-GAN like architecture seems to be that they have an ensemble of discriminators, and make them into a (n + 1) classifier to support n-class classifier (instead of binary as a the original gan paper).

# [ Erasing Concepts from Diffusion Models ](https://arxiv.org/pdf/2303.07345.pdf)
[Website](https://erasing.baulab.info/)

- Using the models own understanding of the concept it can be used to erase the concept. This is done by giving the model the text description of the concept.
- This can be done by querying a frozen model to generate noise for a given concept, and then train our model to go to the opposite direction of the concept.

### [Consistency Models](https://arxiv.org/pdf/2303.01469.pdf)

Consistency models are a single step generation which is cool.

Tries to be an alternative to Diffusion models by being fast. Part of the math used goes above my head on this first read, but sounds like the general idea is to learn an universal mapping from any point to the origin. The the algorithm they have makes sense, but that seems to be multistep though.

Algorithm 2 and algorithm 3 need to be reviewed a bit more. They use a ODE solver though while training.

The model output is also very nice.

TODO, look at the code they released [Consistency Models(code)](https://github.com/openai/consistency_models)

Implementation by [others](https://twitter.com/RiversHaveWings/status/1634038603247661062) also shows that it seems promising. 
This is an alternative [open source](https://github.com/cloneofsimo/consistency_models) version.


### [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/data/paper.pdf)
It's based on the StyleGAN2 architecture. This paper was a bit harder to digest(maybe I need more coffee), but as I understand it is the "Motion supervision" that is the core part of the loss. The idea seem to be that you just move the point a bit, and make sure the model generates something similar. The other key part seems to be the way this point tracking of the loss is done.
-> Fig. 4 the car is switched out, but the person and the sunset seems to be mostly the same. Interesting.
-> 
[code (will be released in june)](https://github.com/XingangPan/DragGAN)

[code (unofficial)](https://github.com/JiauZhang/DragGAN)

[Good tweet on draggan](https://twitter.com/mayfer/status/1659940842965200901?s=12)

### [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)
Found by this [tweet](https://twitter.com/iscienceluvr/status/1661565298066198536?s=12).  

By reading the paper it also sounds like a form of RLHF.

Pipeline
-> Prompt into diffusion model
-> Output image to image 2 text model
-> Bert similiarty score between prompt and image caption.

### [CoDi: Any-to-Any Generation via Composable Diffusion](https://arxiv.org/pdf/2305.11846.pdf)
Cool paper, allows you to get multiple modalities from other modalities.  
-> Text, image, audio -> text, image, audio

The name "Composable Diffusion". Sounds like how it is solved is by having all the modalities are into the same latent space. 


Side note: I think' I read a paper similar to this, where they just use special tokens for other modalities and pushed it into a transformer.

### [Self-Consuming Generative Models Go MAD](https://arxiv.org/pdf/2307.01850.pdf)

They basically feed a GAN it's own output and see what happens. 
The conclusion is that you need a certain amount of fresh/real data to not have the model degrade. You can't have model generation on only GAN output.
