## [Learning skillfull medium-range global weather forecasting](file:///Users/brage/Downloads/science.adi2336.pdf)
[Source code](https://github.com/google-deepmind/graphcast)

- Improved machine learning model by using historical data
  -  Allegedly based on what is said in teh paper - traditional methods uses more compute resources for greater predictions 
  -  https://en.wikipedia.org/wiki/Numerical_weather_prediction
-  Current methods often uses supercomputers while GraphCast can use a single Google TPU v4
-  The model itself 
   - Autoregressive -> Can be rolled out based on its own inputs
   - Input is two two weather states and predicts the 
-  Trained for 4 weeks with 32 Google TPU v4

## [Deep Reinforcement Fuzzing](https://arxiv.org/abs/1801.04589)
Uses Q-learning as a way to do fuzzing. There isn't much more to say, but the idea is quite cool as generally fuzzing algorithms use something like evolutionary algorithms.

