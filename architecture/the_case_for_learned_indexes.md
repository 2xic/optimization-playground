## [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)
Uses machine learning for data structures like hashtables, b-trees and bloom filters.
They use a hierarchy of models to be able to predict the position. 

The key idea is more or less that the models learn a CDF

$$p=\mathcal{F}(key)\cdot N$$

where N is the total numbers of keys and `F` is the estimated cumulative distributing. 


[ Stanford Seminar - The Case for Learned Index Structures ](https://www.youtube.com/watch?v=NaqJO7rrXy0) with a talk from one of the Authors.
