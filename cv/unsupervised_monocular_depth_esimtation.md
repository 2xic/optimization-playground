[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf)

- Current methods used supervised learning (usually), which requirers a lot of ground truth
- Instead we use binocular stereo footage!
- Other approaches
  - structure from motion
  - shape-from-X
  - binocular
  - multi-view stero


Method
- Use two images corresponding to left and right color images from calibrated camera pair
  - Goal is to find dense that correspondence fields that when applied to the left image would enable us to reconstruct the right image
  - The same should be true for the other way also! I.e apply d to right and get left
- Loss has four output scales (EQ 1 / FIG 2)
- 


----
I think we will try to implement this, use single camera, but with video.
Should be possible to model stero with single camera + time

