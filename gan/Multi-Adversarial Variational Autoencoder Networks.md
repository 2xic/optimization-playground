### [Multi-Adversarial Variational Autoencoder Networks](https://arxiv.org/pdf/1906.06430.pdf)
MAVEN attempts to be a more robust GAN aritecture. They use 3 networks, encoder, generator and a discrimnator.
The job of the encoder is to compress the input to a lower dimension, the job of the generator is to generate inputs to trick the discrimantor, and the discrionators job is to learn what is real and fake.
The main difference between this aritecture from a traidional GAN / VAE-GAN like aritecture seems to be that they have an essemble of discrionmantors, and make them into a (n + 1) classifier to support n-class clasifier (instead of binary as a the original gan paper).
