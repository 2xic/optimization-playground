# SimCLR
Speedrunning paper and testing PyTorch Lightning.

I think I have implemented most things correctly, but not able to get the training to converge currently.

[paper](https://arxiv.org/abs/2002.05709)

[PyTorch Lightning](https://www.pytorchlightning.ai/)

## Reflection
- The loss was wrong and caused issue. This was not caught by the tests. Not good.  
  - Debugging against https://theaisummer.com/simclr/ helped find the bug.
