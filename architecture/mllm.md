# [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045.pdf)
*Fuses vision models and large language models** and the model (KOSMOS-1) does well on several tests related to language understanding, generation, perception language tasks, and image recognition… 
I would also add they have several cool examples in the paper.

So the authors take the approach that the transformer is a general purpose interface, so the way "multimedia" is encoded is by having tags like `<image>` amd `<document>` to guide the model. Separate models are used to get the embeddings the image / documents.

They have the traditional training objective of predicting the next token given a context. 


### [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)
This paper influenced [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045.pdf).
The idea is to use transformer as an interface to various other foundation models. 
