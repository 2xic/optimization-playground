### [Solving rubicks cube with robotic hand](https://arxiv.org/pdf/1910.07113.pdf)
The core idea here is automatic domain randomization which gradually generates more complicated environments. This allows `sim2real` to work very well.
They use `MuJoCo` for the physical system, and `ORRB` is used is used to render synthetic images for the CV.

ADR is explained in section 5 with an attached algorithm description. The core idea is to have a method of having the environment becoming more complicated as the model becomes smarter.
Things that are randomized
- Cube size
- Friction
- Action delay
- Action latency
- Gravity
- 