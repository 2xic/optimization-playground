[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

The paper looks at how balanced network depth, breadth, and resolution can lead to improved results. They propose a new scaling method based on the findings.
The proposed methods scales network depth, breadth and resolution.

$$\text{depth}, d = \alpha^\phi$$
$$\text{width}, w = \beta^\phi$$
$$\text{resolution}, r = \gamma^\phi$$
Where $\phi$ control's how much resources are available, and $\alpha, \beta, \gamma$ is controlling how to scale the resources and can be determined by a grid search on the base model.

Such that

$$\alpha \times \beta^2 \times \gamma^2 \approx 2$$
$$\alpha \ge 1, \beta \ge 1, \gamma \ge 1$$

