Sometimes the solution to a problem is just a cleaver pipeline (and not [messy pipelines](https://imgs.xkcd.com/comics/machine_learning_2x.png)).

## [Learning the 3D Fauna of the Web](https://arxiv.org/pdf/2401.02400.pdf)
[Demo](https://huggingface.co/spaces/Kyle-Liz/3DFauna_demo)

[Website](https://kyleleey.github.io/3DFauna/)

The pipeline is something like this
1. First encode the image by using a image encoder (pre-trained unsupervised)
2. This output is then used 
   1. to query a shape bank and get the [DINO](https://towardsdatascience.com/dino-emerging-properties-in-self-supervised-vision-transformers-summary-ab91df82cc3c) features
   2. to predict the pose, shading, etc of the model
3. Then a re-render is preformed based on the above information
4. THe loss is then the reconstructed error vs the input image + discriminator loss
