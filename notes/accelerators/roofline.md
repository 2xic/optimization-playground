## roofline plots

### TLDR
$$\text{FLOPS} = \frac{\text{#flop}}{\text{time (sec)}} = \frac{\text{#Flop}}{\text{Byte}} \cdot \frac{\text{byte}}{sec} = \text{Artimetic instenity (AI)} \cdot  \text{bandwith (BW)}$$

`Arithmetic Intensity` is the number of operation per data unit. 

Roofline is taking the $log(flops)$ against $log(AI)$


### Some good getting started content
- [Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [Understanding Roofline Charts](https://www.telesens.co/2018/07/26/understanding-roofline-charts/)
