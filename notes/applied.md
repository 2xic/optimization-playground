## [Millions of new materials discovered with deep learning](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/)
- [This professor doesn't like it](https://twitter.com/Robert_Palgrave/status/1730358675523424344)
- 

## [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)
[source code](https://github.com/google-deepmind/alphafold)

### AlphaFold 3
- [Thread](https://x.com/jankosinski/status/1788283743423045642)
- [Blogpost](https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model/)
- [Paper](https://www.nature.com/articles/s41586-024-07487-w)
- It's a collab with [isomorphic labs](https://www.isomorphiclabs.com/articles/alphafold-3-predicts-the-structure-and-interactions-of-all-of-lifes-molecules)
- 

*todo*

## NeuralHash
- Apple made it for some of their [safety features](https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf)
- TLDR: Similar looking images will have same neural hash. i.e changes to aspect ratio doesn't change the hash and same is true for colors etc.
- The pipeline is like this
  - Image -> Feature encodings -> Hashing based on local-sensitives hash
- [Adversarial attacks](https://towardsdatascience.com/apples-neuralhash-how-it-works-and-ways-to-break-it-577d1edc9838) can beat the system. [Source code](https://github.com/greentfrapp/apple-neuralhash-attack)
- More attacks on [neural hash](https://jagdeepsb.github.io/assets/pdf/neuralhash%20icml%20ml4cyber%202022.pdf)
- [Getting the model](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX?tab=readme-ov-file)
- 