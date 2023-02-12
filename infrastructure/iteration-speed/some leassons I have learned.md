#### Some lessons I have learned
- Visualize everything, it's always easier to spot mistakes this way.
- Test what is possible to test and it hard to visualize
- Make it easy to get GPUs
  - https://github.com/2xic/lambda-labs-api
- Make it easy to use the GPU/CPU power.
  - Resource usage should be utilized to 100%

### Bugs I have ran into
- Bugs I have ran into, which could have been solved by better tests
  - Using eval when I was supposed to use no_grad
  - Bug in dataloader causing it to have wrong __len__ causing bugs in evaluation metrics
  - Randn vs rand
