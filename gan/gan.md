# [Scaling up GANs for Text-to-Image Synthesis](https://arxiv.org/pdf/2303.05511.pdf)
GigaGan is an attempt to level up GANs so they can "combat" against the new hot diffusion models, they tried scaling up StyleGan, but that did not work, so they ended up creating a new architecture. 
*The results looks quite nice.*
They have a few ideas that I have not seen before related to GANs, one is the usage of CLIP as a pretrained model for the text encoding. They also use a bank of convolution filters to not have to scale up the width of the convolution layers.

# [Cones: Concept Neurons in Diffusion Models for Customized Generation](https://arxiv.org/pdf/2303.05125.pdf)
This paper finds that certain neurons in a diffusion model maps to certain subjects (called concept neurons in the paper), they can be identified by the statistics of network gradients for a given subject.
The paper tries to explore ways to control the concept neurons to generate a subject in diverse settings.
They also propose a algorithm for finding concepts neurons, so if by scaling down a neuron increases it's loss it would be regarded as a concept neuron.
They also show it's possible to combine multiple concepts into a single image, and the results look quite good.
 

