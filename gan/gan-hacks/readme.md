Not an hack, but looking at various discussions for how to train stable GANS.

Note that this Gan is of the simpler type, but will add more advanced architecture to test soon, and maybe compare them under certain conditions.

### ["How to Train a GAN? Tips and tricks to make GANs work"](https://github.com/soumith/ganhacks) 
1. Normalize the inputs, and use tanh as the output layer of the generator.
2. Do not use the traditional min-max loss
    - Also mentioned in [https://developers.google.com/machine-learning/gan/loss](https://developers.google.com/machine-learning/gan/loss) and [https://github.com/tensorflow/tensorflow/blob/2007e1ba474030fcce840b0b8a599558e7d5998f/tensorflow/contrib/gan/python/losses/python/losses_impl.py#L412](https://github.com/tensorflow/tensorflow/blob/2007e1ba474030fcce840b0b8a599558e7d5998f/tensorflow/contrib/gan/python/losses/python/losses_impl.py#L412)
3. Avoid layers and activations that results in sparse gradients.
    - i.e relu or maxpool
4. Tracking / noticing failures
    - If things goes smoothy you should notice the following
        - No spikes in the gradients
        - The loss of the generator should however not decrease smootly (if it is, it might be fooling the discriminator)
5. Apply noise to inputs to discriminator
6. Use dropout in the generator
    - This also was a nice help for me
7. The kind of distribution you use matters.
    - I have also noticed this
8. Use labels
    - **I notice this also, GANs without labels are not very good, I only get some kind of mode collapse**

[Tips On Training Your GANs Faster and Achieve Better Results](https://medium.com/intel-student-ambassadors/tips-on-training-your-gans-faster-and-achieve-better-results-9200354acaa5)
1. Size of input noise effects the performance of the model (using 100 might not be the best idea)
    - I was not able to see any effect based on this
2. Using lower batch sizes might also be smart.
    - This helped some actually
