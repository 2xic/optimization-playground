EPOCHS = 100
BATCH_SIZE = 128

def set_no_grad(model):
    for name, param in model.named_parameters():
        #if not "fc2" in name:
        param.requires_grad = False
