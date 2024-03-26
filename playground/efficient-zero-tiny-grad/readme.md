# [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf)

Mostly to play around with tinygrad.

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```


### Run tests
```bash
python3 tests.py
```

### Testing with pytorch
```bash
python3 train.py pytorch # 2.6573272049427032, 2.6666435545141045, 2.6724558612879585
```


### Testing with tinygrad
It's slower than torch.

```bash
# Running this is super slow
# python3 train.py tinygrad # -> 20.957842111587524, 21.923494577407837, 19.40097141265869

CPU=1 python3 train.py tinygrad # -> 12.663580656051636, 10.755399942398071, 13.984618902206421
```


### TLDR:
- Builds upon Muzero ideas
- Uses [simsam](https://arxiv.org/pdf/2011.10566.pdf) for contrastive loss

### TODO
- [X] Complete the monte carlo logic
- [ ] Get the model working on the simple RL environment

