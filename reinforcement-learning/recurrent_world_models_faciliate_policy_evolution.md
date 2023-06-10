# [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/pdf/1809.01999.pdf)

- RNN is trained in a unsupervised manner to model popular reinforcement learning environment through compressed spatiotemporal representation
  - Humans learn abstract representation of the world
    - Our actions are influenced by our internal predictive model.
      -  For many problems, an RL agent might also benefit from a predictive model 
- The world models extracted features are fed into compact and simple policies trained by evolution

- The world model will be a large RNN that learns to predict the future given the past in an unsupervised manner
    - Past observations and actions are perceived and exploited by another NN (controller) which learn through RL to preform some task without a teacher
      - 

## Layout
- Visual component
- Memory component (Predictive)
- Controller

            v--------------------------------
- Environment -> VAE                        |
                     -> Controller ---> ACtion  
                 RNN ---^                   |
                  ^-------------------------v

- The role of the VAE is to compress the world.
- RNN remembers the action history
  - Predict model future of the compressed world as an probability distribution (mixture Gaussian distribution)
- Controller does the action
  - Should be simple and trained separately
- 


-----
My notes
- 
- Reminds me somewhat of https://arxiv.org/pdf/1608.01230.pdf
