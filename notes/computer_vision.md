## Computer vision
For the paper notes that are so short that they don't need their own file.

### [Segment Anything](https://scontent.fosl3-1.fna.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5R36a2pHTEoAX--qhR_&_nc_ht=scontent.fosl3-1.fna&oh=00_AfB90Y1aIGlaPf2UpYj8__m5_7ivPl9U086PrP4B3CKO-A&oe=6440DE67)
Glanced of the paper and it seems interesting

- Goal was to build a new foundation model with required three components
  - Prompt for segmentation task (which can be any text that can indicate what to make a segment for)
  - Segmentation model
  - Data engine for all the data
- Core model components
  - Image goes into an image encoder
  - The embeddings can then be used as query for the prompt
  - The model can output multiple results if the input is ambiguous and give them a score

[Blogpost by meta](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)

## [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf)
- Self supervised and it requirers no fine tuning. 
- By being self supervisied the model can be more "powerful", classical models that use train models with captions usually have some missing information. It also allows it to operate in domains with few expert since the data does not need to be annotated.
- The goal is to learn a visual vector.
- They have a good data pipeline
- Discriminative self-supervised methods
  - Image-level objective -> Cross entropy loss between student and teacher model
  - Patch-level objective -> Patch part of the image to the student, but not the teacher and check cross entropy loss
  - In addition to several other methods
- They use model distillation to 

[Blogpost by meta](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/?utm_source=twitter&utm_medium=organic_social&utm_campaign=dinov2&utm_content=video)

## [What does CLIP know about a red circle?](https://arxiv.org/pdf/2304.06712.pdf)
**By drawing red circles, CLIP will put more focus on that and it works as a visual way of doing prompt engineering**

By drawing red circles, and prompting the model with "this is *animal*" the model should give the highest accuracy on the correct model. This seems to work well in practice (see figure 2).

### [ImageBind: Holistic AI learning across six modalities](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)
They train one model to lean the image mapping, and one model to convert from another domain into a representation within that domain. Two networks that are jointly trained similar to CLIP.

### [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017)
CLIP is cool, but it required a lot of recourses for training. The authors of this paper have however figured out a way to make this barrier to entry a lot smaller.

What they figured out was that by using a larger text / image encoder, the sequence length can be smaller when training (allowing faster training). In other words, less text tokens, and smaller image.

Thereby the name "inverse scaling law". By using a larger model, you can decrease the input size, and still get the same or close to the same results.

### [Rosetta Neurons: Mining the Common Units in a Model Zoo](https://arxiv.org/pdf/2306.09346.pdf)
Question: Does computer visions models share some common representation ? 
Answer: Yes based on the investigations done by the authors they do.

In short the method works by comparing activations maps between models. Some activations maps might have different sizes and to deal with that they do a bilinear resizing of the smallest activation map.

