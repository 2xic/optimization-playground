
def get_parameter_count(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    return total_params
