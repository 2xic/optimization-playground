### [Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignmen](https://arxiv.org/pdf/2208.13628.pdf)
Vision and language pretraining usually consist of simple alignment tasks like image and text alignment (CLIP). Since there is a lot of data for this kind of task on the internet, this has gained a lot of attention.

The authors propose a new framework for doing this more efficiency consisting of a vision encoder, and a text encoder that both are aligned through a "Hierarchical Image-Text Contrastive block". The image is also fed into a "Visual Concepts Module" that fetches additional visual features for the model by for instance using CLIP which is then fed into the Vision encoder.
Both the Vision encoder and Text encoder feed the output into a Multimodal Decoder that learns the relationship.
The visual concepts module is meant to help guide the visual encoder, and sounds like it's partially inspired by how Text promps are used to guide LLM. 
There are a few methods they try to get the best performance (they get the best with all of them enabled)
- Hierarchical Image-Text Contrastive block, special layer that is meant to make the alignment of the vision model and the text model happen from the start, and not at the last stage.
- Image-Text matching, binary cross entropy loss to see if image and text matches
- Masked language modeling, predict a masked token with context
- Masked image modeling, mask part of the image, and try ot reconstruct.
- Unimodal image modeling masked image modeling, they also have a small vision decoder as part of the framework that takes the output tokens of the visual component with masked tokens, and learn to reconstruct the image.
- Multimodal masked image modeling, now the model also have access to textual tokens to help with the reconstruction. 

### [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://open-eqa.github.io/assets/pdfs/paper.pdf)
[Blog post](https://ai.meta.com/blog/openeqa-embodied-question-answering-robotics-ar-glasses/)
- New benchmark for evaluating models. Testing episodic memory and contextual understanding of both text and video.
- 
