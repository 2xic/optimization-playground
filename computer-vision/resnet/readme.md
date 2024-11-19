## Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385.pdf


### Results

|                 Testing accuracy                 |                 Training accuracy                 |                 Loss                 |
| :----------------------------------------------: | :-----------------------------------------------: | :----------------------------------: |
| ![](./testing_accuracy_resnet_vs_non_resnet.png) | ![](./training_accuracy_resnet_vs_non_resnet.png) | ![](./loss_resnet_vs_non_resnet.png) |


## Plot curve
I tried to reproduce the loss landscape with / without skip connections ([paper](https://arxiv.org/pdf/1712.09913)) 

Didn't fully work, but you get the idea.

|         No skip connection          |         Skip connection          |
| :---------------------------------: | :------------------------------: |
| ![](./loss_no_skip_connections.png) | ![](./loss_skip_connections.png) |

