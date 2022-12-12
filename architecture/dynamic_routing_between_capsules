[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

- Capsule: group of neurons whose activity vector represents the insanitation parameter of a specific entity such as a object, or an object part  
    - Vector length is used to represent the probability that the entity exists
    - Vector orientation is used to represent the insanitation parameters
- Transformation matrices are used to make prediction at one layer for higher level capsule
  - When multiple capsules agree, a higher level capsule become active
- squashing is used ensure output vectors can represent a probability (eq. 1)
- Input to an capsule is the weighed sum over all prediction vectors
  - Prediction vectors are capsule output multiplied by a weight matrix (eq. 2)
    - This is true for all layers except the first
    - 
- I guess the core idea here is routing mechanism allowing the network to stich information together in a more unique way.


This blog had some good notes [“Understanding Dynamic Routing between Capsules (Capsule Networks)”](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/).
