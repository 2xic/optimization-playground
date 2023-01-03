## Notes from the paper

- Random masks applied to image to better understand what the model finds important
- Black-box approached
- Mask * the image -> figure out what is important

Mask generation
- f(mask * image)
  - Importance of a pixel is the expected score over all possible masks condition on that pixel being observed
- See section 3.2

