### [Learning a Driving Simulator ](https://www.youtube.com/watch?v=hpRzNxQvZDI)


## Old: small offset simulator
Shift the image in different directions to simulate movement in the image.

This "simple" model is good enought to 
- learning to recover from drifts away from the land line
- stop for stop signs
- stop for red lights

[mentioned here](https://blog.comma.ai/end-to-end-lateral-planning/)

## why the need to improve ? 
The model starts to cheat by using simulator artifacts. Small offset don't work well with larger offset.

## evolution  
- Image tokenizer
  - Fancy image compressor
  - Image to token
  - Image encdoer goes from image to tokens
  - Image decoder goes from tokens to image
- Pose tokenizer
  - Tokenize the value of the [car pose](https://github.com/commaai/openpilot/blob/79a6512345e80269050ab4fb753564c3d0f0ebe2/common/transformations/README.md)
- Dynamic transformer
  - Autoregressive sampling
    - One token at the time to generate the next input

This setups works! It's also able to capture things like accelerations and breaking. 

## some side notes
- They use a recurrent layer for more temporal information between frames

