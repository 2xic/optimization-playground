## Openpose

https://arxiv.org/pdf/1812.08008.pdf


# TODO
- Optimize the L_c generation
- Fixed the skeleton plotting
  - There is something wrong in the way I do non maximum suppression
  - Should correctly assign for each person
    - Try to plot all predictions for a limb without a max supression
    - vs the actual value
    - Probably the issue is that sigma is to large.