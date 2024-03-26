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

### [Dynamic Handover: Throw and Catch with Bimanual Hands](https://arxiv.org/pdf/2309.05655.pdf)
[Project page](https://binghao-huang.github.io/dynamic_handover/)
- System trained in simulation (IsaacGym physical simulator) and then sim2real is applied
- Neural network used as a goal estimator to be abel to remove some of the gaps between the simulation and the real world

### [Twisting Lids Off with Two Hands](https://arxiv.org/pdf/2403.02338.pdf)
- They don't use raw image pixels as inputs, but instead use the Segment Anything mode and XMem to track the segment mask through the remaining
- They also use RL ofc. I find the reward function a bit complicated and not as clean as I hoped.
  - They apply Domain Randomization also
  - Sim2Real

### [Universal Manipulation Interface](https://umi-gripper.github.io/)
Very cool they open sourced both the software and the hardware they used. Sounds like they use slam and they also have a [diffusion model](https://github.com/real-stanford/universal_manipulation_interface/blob/e02f7a960fef9b529c0af10d6452072cfd53819f/diffusion_policy/model/vision/timm_obs_encoder.py#L17) for the policy. The videos make it seem impressive. 

Very similar to this paper - [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/aloha.pdf)
https://tonyzhaozh.github.io/aloha/. However the execution here is a lot better.


