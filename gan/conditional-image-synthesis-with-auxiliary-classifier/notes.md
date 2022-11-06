# Notes from the paper

- Label conditioning on sampling
- The GAN framework can be augmented using side information.
  - One strategy is to supply both the generator, and the discriminator with class labels in order to produce class conditional samples.
  - *Instead of feeding side information into the discriminator, one can task the discriminator with reconstructing the side information.*
  - This is done by modifying the discriminator to contain auxiliary decoder network that outputs the training label
  
### AC-GAN
- Every generated sample has a corresponding label, this is in addition to the noise latent variable.
  - The generator would use both variables to generate a sample
- The discriminator is given both probability distributions over the class labels
  - P(S | X) <- source of x
  - P(C | X) <- class of x
- The objective of the discriminator is to identify the correct source, and the correct class.
  - L_s = log(P(S = real | X = real)) + log (S=fake | X = fake)
  - L_c = log(P(C = c | X_real)) + log(P(C = c | X = fake))
  - Discriminator want to maximize  L_s + L_c
  - While generator want to maximize L_C - L_s

### Generating high resolution images improves discriminability
- Goal of an image synthesis model should be to produce high resolution images that are more discriminative than low resolution images
- 




