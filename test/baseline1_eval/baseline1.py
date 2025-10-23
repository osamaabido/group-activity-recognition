import os 
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)


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
        num_classes=config['model']['num_classes']
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
        videos_path=f"{config['data']['videos_path']}",
        annot_path=f"{config['data']['annot_path']}",
        split=config['data']['video_splits']['test'],
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
   
    path = rf"{ROOT}\test\baseline1_eval"
    prefix = "Group Activity Baseline 1 eval on testset"

    metrics = model_eval(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        path=path,
        prefix=prefix,
        class_names=config['model']["num_classes_label"]
    )

    return metrics
                                 

if __name__ == "__main__":
    ROOT = "H:\Group-Activity-Recognition\group-activity-recognition" 
    MODEL_CONFIG = rf"{ROOT}\configs\Baseline1.yml"    
    CHECKPOINT_PATH = rf"{ROOT}\test\baseline1_eval\final_model.pth"

    metrics = eval(ROOT, MODEL_CONFIG, CHECKPOINT_PATH)
    print(metrics)
  
"""
  ==================================================
Group Activity Baseline 1 eval on testset
==================================================
Accuracy : 70.28%
Average Loss: 1.4405
F1 Score (Weighted): 0.7028

Classification Report:
              precision    recall  f1-score   support

       r_set       0.69      0.65      0.67      1728
     r_spike       0.78      0.73      0.76      1557
      r-pass       0.60      0.65      0.62      1890
  r_winpoint       0.78      0.75      0.77       783
  l_winpoint       0.78      0.82      0.80       918
      l-pass       0.64      0.64      0.64      2034
     l-spike       0.77      0.82      0.79      1611
       l_set       0.70      0.67      0.68      1512

    accuracy                           0.70     12033
   macro avg       0.72      0.72      0.72     12033
weighted avg       0.70      0.70      0.70     12033
  """