# [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045.pdf)
*Fuses vision models and large language models** and the model (KOSMOS-1) does well on several tests related to language understanding, generation, perception language tasks, and image recognition… 
I would also add they have several cool examples in the paper.

So the authors take the approach that the transformer is a general purpose interface, so the way "multimedia" is encoded is by having tags like `<image>` amd `<document>` to guide the model. Separate models are used to get the embeddings the image / documents.

They have the traditional training objective of predicting the next token given a context. 


### [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)
This paper influenced [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045.pdf).
The idea is to use transformer as an interface to various other foundation models. 

### [Prismer: A Vision-Language Model with An Ensemble of Experts](https://arxiv.org/pdf/2303.02506.pdf)
Uses pretrained "experts" (that are freezed during training), and have only a few components that are needed to train to be able to compete with existing SOTA (and with less data!).
Model architecture is a encoder-decoder transformer, output of a vision encoder is fed into a language decoder conditioned(using cross attention) on the vision features.

Image -> Experts -> Compressed features
Text -> decoder -> P(text | Compressed image features) -> Output

### [A Generalist Agent (GATO)](https://arxiv.org/pdf/2205.06175.pdf)
Model that is capable of playing Atari, captioning images, chatting, and more! This all comes from a single model GATO. The model is trained supervised, but the authors belive it should be possible to train it with RL.
Everything is converted into tokens, and by doing this GATO can be trained like a LLM. 

They mostly show results from the RL, and it has a good success rate in that environment. The paper also have some examples from the other tasks GATO can do, but not that many benchmark results in those domains.



