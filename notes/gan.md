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
Paper is interesting, maybe try to reimplement part of it.

# [Multi-Adversarial Variational Autoencoder Networks](https://arxiv.org/pdf/1906.06430.pdf)
MAVEN attempts to be a more robust GAN architecture. They use 3 networks, encoder, generator and a discriminator.
The job of the encoder is to compress the input to a lower dimension, the job of the generator is to generate inputs to trick the discriminator, and the discriminators job is to learn what is real and fake.
The main difference between this architecture from a traditional GAN / VAE-GAN like architecture seems to be that they have an ensemble of discriminators, and make them into a (n + 1) classifier to support n-class classifier (instead of binary as a the original gan paper).

# [ Erasing Concepts from Diffusion Models ](https://arxiv.org/pdf/2303.07345.pdf)
[Website](https://erasing.baulab.info/)

- Using the models own understanding of the concept it can be used to erase the concept. This is done by giving the model the text description of the concept.
- This can be done by querying a frozen model to generate noise for a given concept, and then train our model to go to the opposite direction of the concept.
- 

