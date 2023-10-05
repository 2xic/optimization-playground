[wavenet](https://github.com/odie2630463/WaveNet/blob/master/model.py)
- skip connections are used in both cases
- likely an issue in the gated connections `v_out` seems wrong on our side


## ideas
- One idea -> is to instead of training it on audio train it on some dummy data and you should still expect to see the model learn if this model is meant to work.
- 
