[Releases notes](https://pytorch.org/get-started/pytorch-2.0/?utm_source=twitter&utm_medium=organic_social&utm_campaign=pytorch_conference)

- Backwards compatible

## Torch.compile()
[docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
[blog](https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/?utm_source=twitter&utm_medium=organic_social&utm_campaign=pytorch_conference)]

- Uses jit to speed up PyTorch code
- Dev team expects a 30%- 2x speed up 
  - They also prove that the changes does a speedup in the release notes
- Builds on other new technologies TorchDynamo, AOTAutograd, PrimTorch, and TorchInductor
    - Theses are all written in python
    - TorchDynamo
      - Acquiring graphs reliably and fast
    - TorchInductor
      - Fast codegen (define-by-run IR)
    - AOTAutograd
      - To capture backpropagation
      - Stands on top of the autograd system
    - PrimTorch
      - PyTorch has 12000 + operators and 2000 + 
        - ???
- 



