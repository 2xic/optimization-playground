### [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/pdf/2207.08815.pdf)

First heard about the paper on the [numerai Quant club](https://www.youtube.com/watch?v=w8Y7hY05z7k) where for instance mentioned that one of the reasons for this phenomena is that NN usually create smoother functions than tree based models.
-> Is this a property that emerges because NNs are differentiable ? 

The goal of the paper is to run evaluation of NNs vs tree based models on various tabular data. While NN have had a huge success on text and image dataset, it's unclear how well they do in tabular data (side note here, it's known meme/fact that kaggle competitions usually are not won by fancy NNs, but instead some variant of Xgboost)
Tree models remains state of the art on dataset of ~10K samples and that is without accounting for the huge speed improvement they also have.The authors have created a benchmark to allow further investigations of why there is this gap between NNs and tree based models.

-> HM, creating tabular-specific deep learning architectures actually seems to undergo very active research. Interesting. However, based on what the authors write here some of the new architectures that claim to beat tree models usually do not do well on new datasets and simple resnet models seems to be competitive against them. However, part of the problem here might be that there is no good benchmarking dataset.

-> [code](https://github.com/LeoGrin/tabular-benchmark) is open source

Looking at the benchmark results, MLP and Resnet get's crushed on classification and regression tasks for tabular data. This is true for numerical features, and both numerical and categorical features.
They also tried to use a [FT_Transformer](https://paperswithcode.com/method/ft-transformer), and [SAINT](https://paperswithcode.com/method/saint), but they also lost against tree based models.

-> Even with tuning of the hyperparameter, NN does not become close to state of the art. 

Why do tree-based model outperform deep learning models ? 
-> So given that hyperparameter does not make NN any more powerful, it gives the suggestion that there is something inherit to the model architectures
-> So one the findings which where also mentioned in the Numerai quant club is that NN are biased to smooth solutions. 
-> Uninformative features affects NN like architectures. They see that removing / adding features of importance affects the gap between NN like architectures and tree based models
    -> Run a an experiment to test this, if this is the case that would be super interesting.
-> "invariant by rotation" (unitary matrix applied to the training and testing dataset). 

In the appendix one can see that with large dataset the SAINT and FT Transformer model starts to be able to compete with the XGBoost like models. This is both for numerical and categorical features. 

