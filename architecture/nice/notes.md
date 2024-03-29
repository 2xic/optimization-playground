## Paper notes from NICE

- Good representation is one which data has a distribution that is easy to model
- Non-linear transformation is learned to make the transformed data conform to factorized distributions


The job of the learner is to learn
- h = f(X)
  - Transformation
- components h_d are independent
  - PH(h) = mul (ph_d(h_d)) for all d
- Inference
  - log(P_x(x)) = log(P_h(f(x))) + det(jaboian x)
- P_h is a predefined density function (i.e gaussian) 
- 
----

----

Transformations
- We want to learn the distributions through a simpler distributions
- Coupling layers
  - Core idea is that X can be split into two parts (X1, X2)
    - (X, l_i, l_2)
  - m is function mapper
  - we can then define (y_1, y_2) as
  - y_1 = x_1
  - y_2 = g(
      X_i_2,
      M(X_i_1)
    ) -> g is coupling law
    - coupling law
      - g(a;b) = a + b
    -     

----

Some good notes
- https://stats.stackexchange.com/questions/493863/inference-in-normalizing-flow-model-nicenon-linear-independent-components-esti
- 