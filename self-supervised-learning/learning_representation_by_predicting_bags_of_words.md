# [Learning Representations by Predicting Bags of Visual Words](https://arxiv.org/pdf/2002.12247.pdf)
Interesting idea similar to that of BYOL (but this paper was published first). 

You have two neural networks with three and two components. 
The first neural network is pretrained, it has an encoder for the feature map, that is passed in a quantization component to get the "visual vocabulary", which is then passed to a BoW encoder component.
The second neural network also has the encoder component, but the next component is the BoW encoder component.
The loss is the cross entropy between the two BoW distributions between the two networks. The second network will receive a permuted version of the image that the first network receives. 

