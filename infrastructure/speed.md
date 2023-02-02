[Development speed over everything - blog.comma.ai](https://blog.comma.ai/dev-speed/?utm_source=pocket_saves)
- Keep the software stack simple, try to avoid complexity, and have this as a "North Star".
- Keep yourself in check, and protect against disaster by having continuos integration
- If it takes more than a week to train a model, it will always take a week + to ship a improved model.
  - Comma therefore enforce 24 hour limit on model training
  - They also have test suites that (based on my interpolation) runs while training to make sure the model converges to something reasonable
- Going from idea/changes to see effect of changes should be minimal


#### Some lessons I have learned
- Visualize everything, it's always easier to spot mistakes this way
- Test what is possible to test and it hard to visualize
- Make it easy to get GPUs
  - https://github.com/2xic/lambda-labs-api

