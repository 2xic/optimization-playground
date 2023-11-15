Dataset from https://github.com/commaai/speedchallenge/tree/master

I just want to train a model on the dataset and see what happens.



## Dataset
20 frames video -> I prefer images

```
ffmpeg -i input.mp4 -vf fps=20 out%d.png
```
