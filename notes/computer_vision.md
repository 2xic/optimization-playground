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
