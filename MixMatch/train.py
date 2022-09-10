
"""
1. Run augmentation on a batch
    - Add a test util to the optimization library maybe
2. Feed the model the augmentations
    - Get the average predictions.
    - Sharpen it
3. Create a batch W of labeled and unlabeled batch
    - MixUp labeled batch
    - MixUp unlabeled batch
4. Loss is applied to the Created batches weighted on lambda.
    - Cross entropy on labeled batch
    - Squared l2 loss on unlabeled batch
"""
