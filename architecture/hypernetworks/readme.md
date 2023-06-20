### [HyperNetworks](https://arxiv.org/pdf/1609.09106.pdf)

The idea is to use one network (hypernetwork) to generate the weights for another network (main network).

Note that the hypernetwork is usually smaller than the main network in size.
The input to the hypernetwork is the structure of the weights (index and size).

### Method
So we know that convolution kernels have weights and filters. Which can be described by a N_in X N_out matrix.
This can be breaded down into a `N_in` matrix with multiple slices. 

So the method
- input is weight description into a embedding layer and is then projected into `n_in` inputs


```
Z = input embedding

a_i = Weights (z) + b
K_i = W_out dot a + B
K_concat = (k_1, k_2...., k_n)
```


### References
[https://en.wikipedia.org/wiki/HyperNEAT](https://en.wikipedia.org/wiki/HyperNEAT)

### Looking at code
Does not seem to be an official implementation, but found one on [github](https://github.com/g1910/HyperNetworks/blob/master/hypernetwork_modules.py)
So there seem to be one misconception that I had, I thought I had to train the HyperNetwork separately, but it's actually just one training loop.


