
- Alternatives to using Bayesian NN
  - With very little modifications!
- NN are bad at quantifying predictive uncertainty
  - MC-dropout
    - Check out this idea, monte carlo dropout

Proposed recipe
1. Use a proper scoring rule as training criterion
2. Use adversarial training
3. Train an ensemble

- Proper scoring rules ?
  - Softmax cross entropy loss is a good one
  - MSE is not good!
    - So use two variables if working with regression, and do something like maximum a posterior.

- Adversarial smoothing ?
  - Fast gradient
- 

The algorithm
- Train n models
  - Mini batch
  - Generate adversarial example for the batch
  - Loss for mini batch + adversarial batch
- Output is combined results of models
- 