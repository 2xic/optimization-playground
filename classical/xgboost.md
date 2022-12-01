[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)

- Gradient tree boosting
    - State of the art results on many standard classification benchmark
- Tree boosting in a nutshell   
  - Dataset D with n examples and m features
  - Tree ensemble model uses K additive functions to predict the output
    - y_i = \omega(x_i)
        - = \sum_1_{k} f_k(x_i)
        - where f(x) = w_q(x) 
        - is the space of regression trees
          - Unlike decisions trees, each regression tree contain a continuos score of each of the leaf
  - L(\omega) = sum(loss(predicted)) + sum(Omega(f_k))
    - Omega
      - penalizes the complexity of the model
  - Gradient ree boosting
    - Scoring functions and loss are defined here
- Other methods used to improve model (in addition to regularized objective)
  - Shrinkage
  - Column subsample
  - 
- Split finding algorithms
  - Key problem in tree learning is to find the best split
  - 

- TODO read the code https://github.com/dmlc/xgboost 
- TODO implement tree boosting

