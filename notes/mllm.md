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

### [SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs](https://arxiv.org/pdf/2306.17842.pdf)
This is an interesting paper
- Image encoder creates a `LLM codebook`
  - This is a "token pyramid"
- This is then used to decode a new image

The result is quite cool, look at figure 6 and you can see input context of a list of MNIST image and one query is "an image of 1+7" and the output is a mnist image of the number 8.

## [Prompt injection in GPT4-V](https://twitter.com/simonw/status/1712976440969646543)
Use the text from the image as a way to inject instructions into the model.

[This user](https://twitter.com/wunderwuzzi23/status/1712996824364048444) embedded a link into the image which the model happily displayed and then also sent over the chat logs.

## [GPT-4V(ision) system card](https://cdn.openai.com/papers/GPTV_System_Card.pdf)
- Does explore the jailbreaking of this model
- "Disinformation risks" has some interesting examples to say the least ....

Not that useful of an paper (technically speaking and that is just my opinion), but might be interesting for people who want to see some information on the evolution of the model.

[Used to trick people that use GPT for reviewing CVs lol](https://twitter.com/d_feldman/status/1713019158474920321)

### [Design2Code: How Far Are We From Automating Front-End Engineering?](https://arxiv.org/pdf/2403.03163.pdf)
- They constructed a more diverse the dataset versus 
- The results still seem very basic though (Figure 7)
