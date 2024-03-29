## [Data2vec 2.0: Highly efficient self-supervised learning for vision, speech and text ](https://ai.facebook.com/blog/ai-self-supervised-learning-data2vec/?utm_source=pocket_reader)
[Paper](https://arxiv.org/pdf/2212.07525.pdf)

Method that allows more universal supervised learning by being able to learn speech, text and vision unsupervised (unified model architecture).

Seems like the main idea is to have an teacher and student.
The input to the student model is a masked input, and it's meant to predict the output the the teacher model which is given the full input.

I guess this makes sense, since the model would then be able to pick up more than local features. However, I'm also surprise this architecture is able to converge. 

Original Data2vec implementation can be seen here [Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/)


-----

Update: Looked more at it, and it seems like the core idea actually is from BYOP. That said, data2vec has changed various parts of the implementation, but the core idea is from BYOP as far as I can tell (data2vec also cites this paper).

https://arxiv.org/pdf/2006.07733.pdf

