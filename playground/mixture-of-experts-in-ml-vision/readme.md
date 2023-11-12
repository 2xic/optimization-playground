Just some playground example

1. One model trains on Cifar
2. One model trains on Mnist


More a proof of concept than something that is meant to be useful.

You should expect to see something like this at the end
```
Error: 0.4759514629840851, Accuracy: 87.5, Epoch: 9
        cifar (should -> 0) 0.2234962433576584
        mnist (should -> 1) 0.9591125845909119
```

## MOE
They are generally used in LLMs, but the point of this experiment is just to see if I can train a simple gate function for Cifar vs Mnist.

- https://deepgram.com/learn/mixture-of-experts-ml-model-guide
- https://www.cs.toronto.edu/~hinton/csc321/notes/lec15.pdf
- https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1

