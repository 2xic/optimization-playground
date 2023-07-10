
def duplicate_tensor_batch(X, batch_size):
    if X.shape[0] < batch_size:
        return X.repeat((batch_size // X.shape[0], 1) )
    return X
