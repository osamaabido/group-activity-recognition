import os
import yaml 
import pickle
import torch
from datetime import datetime
class Config:
    def __init__(self , config_dict):
        self.model  = config_dict.get("model" , {})
        self.training = config_dict.get("training" , {})
        self.data = config_dict.get("data" , {})
        
    def __repr__(self):
        return f" the config is  model :{self.model} , training : {self.training} , data : {self.data}"
    
def load_config(config_path="config.yaml"):
    with open(config_path , "r") as file:
        config = yaml.safe_load(file)
    #config = Config(config)
    return config

def save_checkpoint_model(model , optimizer , epoch , val_acc , exp_dir , config , best_model = False):
    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'val_acc' : val_acc,
        'exp_dir' : exp_dir,
        'config' : config
    }
    checkpoint_path  = os.path.join(exp_dir ,  f"checkpoint_epoch : {epoch}.pkl")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    if best_model:
        best_model_path = os.path.join(exp_dir , "best_model.pth")
        torch.save(checkpoint , best_model_path)
        print(f"Best model saved at {best_model_path}")
        
def load_checkpoint_model(checkpoint_path , model , optimizer , device):
    """
    Load model and training state from checkpoint
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer
        device: Device to load the model on
    Returns:
        tuple: (model, optimizer, config, exp_dir, start_epoch)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            torch.save(checkpoint, checkpoint_path)    
        except Exception as pickle_error:
            raise RuntimeError(
                f"Failed to load checkpoint using both torch.load and pickle.load: {pickle_error}"
            ) from pickle_error

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer == None: return model
       
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer states to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    start_epoch = checkpoint['epoch'] + 1
    config = checkpoint.get('config', None)
    exp_dir = checkpoint.get('exp_dir', None)
    
    return model, optimizer, config, exp_dir, start_epoch


def setup_training_directories(base_dir='runs'):
    """Create timestamped directory for saving training artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir