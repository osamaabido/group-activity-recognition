import os 
import sys

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from helper_utils import load_config
from eval_utils import model_eval
from dataloader import Group, group_activity_labels
from models import Group_Activity_Classifer



def eval(ROOT , config_path , checkpoint_path):

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Group_Activity_Classifer(
        num_classes=config.model['num_classes']
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_dataset = Group(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels, 
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
   
    path = f"{ROOT}/modeling/baseline 1/outputs"
    prefix = "Group Activity Baseline 1 eval on testset"

    metrics = model_eval(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        path=path,
        prefix=prefix,
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics
                                 

if __name__ == "__main__":
    ROOT = "H:\Group-Activity-Recognition" 
    MODEL_CONFIG = f"{ROOT}\"    
    CHECKPOINT_PATH = f"{ROOT}/modeling/baseline 8/outputs/Baseline_B8_Step_B_V1_2025_01_09_19_29/checkpoint_epoch_63.pkl"

    metrics = eval(ROOT, MODEL_CONFIG, CHECKPOINT_PATH)
    print(metrics)
  