### [Segment Anything](https://scontent.fosl3-1.fna.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5R36a2pHTEoAX--qhR_&_nc_ht=scontent.fosl3-1.fna&oh=00_AfB90Y1aIGlaPf2UpYj8__m5_7ivPl9U086PrP4B3CKO-A&oe=6440DE67)
Glanced of the paper and it seems intresting

- Goal was to build a new foundation model with requiered three components
  - Prompt for segmentation task (which can be any text that can indicate what to make a segment for)
  - Segmentation model
  - Data engine for all the data
- Core model components
  - Image goes into an image encoder
  - The embeddings can then be used as query for the prompt
  - The model can output multiple results if the input is ambigious and give them a score
- 

