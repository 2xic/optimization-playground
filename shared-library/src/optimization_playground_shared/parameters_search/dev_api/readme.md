Thinking about a improved interface

```python
@parameters(x=Range(), y=SearchSpace())
def training_function(x, y):
    pass 
```

1. Define the parameters as a decorator instead of a dict object
2. Having the function hide the for loop complexity

## Optimizing for scale
One issue with the current approach is that it doesn't optimally allocate resources.

- Look at current running experiments and match parameters with the gpu usage
  - This isn't fully accurate as more resources are used on backprop 
  - But I think it could be good enough 

## Goal of the module
1. Doing parameters search at scale efficiently
2. ^ most insights of ML likely is by being able to analyze the effect of parameters at scale and looking at the results

## Starting small and then scaling
1. Could probably get an idea of the effect of an parameter on a smaller dataset and then scale the dataset after initial results
2. 


