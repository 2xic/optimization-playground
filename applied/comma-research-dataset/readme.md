Dataset from https://github.com/commaai/speedchallenge/tree/master

I just want to train a model on the dataset and see what happens.

## Sub-projects
- [Speed challenge](./speed_challenge/)
- [Learning a driving simulator](./learning_a_driving_simulator/)

## Dataset -> Images
20 frames video -> I prefer images

```bash
ffmpeg -i input.mp4 -vf fps=20 out%d.png
```
