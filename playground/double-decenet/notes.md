
## [Tuning tips](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
-  Set num workers for dataloader (done), usually data loader will load synchronously 
- Don't use `zero_grad`, instead set param.grad to none (done? Seems to be the new default)


[Profiler tips](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- Can try to execute this on the code and see if it helps figure out what slow down everything.

[Nvidia presentation of speedups](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf)
```python
torch.backends.cudnn.benchmark = True
```

[Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
- Accumulate gradients

### Tested changes with improvements of speed
Baseline 35 sec, which mean it would take 36 hours to train one of the models, and almost a week to train 4 models.

-> `torch.backends.cudnn.benchmark = True`, seems to be unchanged
-> Update to torch 2, slowed down (10 sec ish).
-> Add `torch.compile` and check the pytorch version, minor speedup.
-> Switch to use `torch.no_grad` in test function, no big change.
-> `set_to_none=True` in the `zero_grad` function (no big change).
-> Accumulate gradients (no big change)
-> [set default device](https://www.learnpytorch.io/pytorch_2_intro/), did not work well
-> Check if both train and test is slow
```
train : 19.764665842056274
testing : 9.0661141872406
```
-> Profiling time spent in forward, it's only 3 seconds about of those 29 seconds. HM.
-> Writing a custom dataloader that does the transform and load everything into memory.     
```
Uses 75 gb for train + test when starting.
```
-> It actually seems to slow down the training loop 4x, what.
-> Same is true for the testing loop, weird.

-> Added a profiler to the loop, and tried to find the bug. So it looks like what take sthe most time is moving to device and calculating the optimize step

-> Huh, halved the batch_size which made it go from 16sec to 3 sec on the move to device.
-> However, now infer and backward suddenly went up.
-> Resizing the images down to 32x32, and trying to increase the batch_size that helped. OpenAI used 128 as batch size though.

-> try to adjust to 16 bit precision ? 
-> try to run [python -m torch.utils.bottleneck ](https://pytorch.org/docs/stable/bottleneck.html)
-> Is the transforms that take time ? 
-> https://discuss.pytorch.org/t/how-improve-the-speed-of-training-after-using-transforms-resize/95994 seems relevant
