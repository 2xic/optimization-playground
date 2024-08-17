import os
import torch

def save_model_atomic(base_name, model):
    temp_path = f"{base_name}_temp.pth"
    final_path = f"{base_name}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
    }, temp_path)
    
    os.rename(temp_path, final_path)

def load_model(base_name):
    final_path = f"{base_name}.pth"
    checkpoint = torch.load(final_path, map_location='cpu')
    return checkpoint
