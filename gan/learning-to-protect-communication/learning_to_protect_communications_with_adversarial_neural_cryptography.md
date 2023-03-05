# [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918)
Goal is to be able to learn ways for two neural networks to communicate without any eavesdroppers being able to decode it.
The networks are trained end-to-end without any specific cryptographic algorithm prescribed.

Homomorphic encryption allows inference on encrypted data (because it's differentiable), but most common algorithms are not differentiable (so it would not be possible to optimize over it).

## Learning symmetric encryption
Goal is to protect plaintext with a shared key.

So Alice and Bob are the two trying to communicate using the shared key, and Eve is the eavesdropper who don't have the key (but is a models?).
Actually all participants are assumed to be neural networks. All communication happens using floating point numbers.

### Objectives
Eve want's to decrypt the message without the key.

Bob want's to read the message.

The loss function of Eve would be to lower the l1 distance of the "Plaintext" and the decrypted version she produced, and she want's to minimize this.

The same loss is applied to Bob, but he also get's the shared key as an input.
The shared loss for Alice and Bob (trained jointly) is the loss between the (Bob_loss - Eve_loss), 

### Architecture
They use "mix & transform" architecture
- One FC layer with the same I/O
    - Plaintext + Shared key is fed into the FC
- Then the next layers are convolution layers 
- The last layer gives an output "suitable for plaintext or cipher-text"
The convolution layers will learn to group some n bits to mix the data. 
An example structure is given in more details in section 2.5.




 







