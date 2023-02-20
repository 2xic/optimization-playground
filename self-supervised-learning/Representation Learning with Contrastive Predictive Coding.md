# [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748v2.pdf)
- Predictive coding -> Based on context and experiences with the current input, what do you expect to happen ? Error is the predicted outcome vs what actually happend
- The method they propose is a bit more complicated
    1. Compress high dimensional data into lantent space so that other models can leavere this information more easily
    2. Use a powerful autoaggresive model (predict next value based on history) 
    3. Noise-Constrastive estimation is used for the loss. I.e use postivie and negative sampels to train model.
    
