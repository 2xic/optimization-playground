## [Object Detection with PCA and Synthetic Data](https://zburkett.io/ai/2023/09/24/pca-object-detection.html)
- First we load a reference image
- Then we load a few guidance images that will be used for the "guidance" and filtering out the dog tokens
- Use DinoV2 to extract the patch features and we are then going to do some PCA analysis on top of that
- Authors of the original paper showed that hte DinoV2 patch features could be used to distinguish between foreground and background by applying some threshold on the patch features. We are going to do something similar. 
- With no guidance the results are quite bad with some guidance, but without the filtered dog tokens the results is better, but still noisy adn finally with both dog tokens filtered and guidance the result is awesome.


