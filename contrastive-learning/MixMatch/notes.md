
[paper](https://arxiv.org/pdf/1905.02249.pdf)

## Summary
By taking the average of `K` predictions of `K` various image augmentation. You sharpen the averaged distribution. 

### Algorithm
- One batch `x` of labeled, and one batch `u` with same size with unlabeled images.
    - Run augmentation on batch `x`
    - Run augmentation on batch `u`
      - Guess a label for `u` by doing an average predictions
        - Sharpe this by doing algorithm 7
      - 
    - MixUp 
      - https://paperswithcode.com/method/mixup
      - 
