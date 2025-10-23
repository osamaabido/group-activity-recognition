import os 
import sys


ROOT = r"H:\Group-Activity-Recognition\group-activity-recognition"
sys.path.append(ROOT)


import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from helper_utils import load_config , load_checkpoint_model
from eval_utils import model_eval
from dataloader import Group, group_activity_labels
from models import Group_Activity_Classifer , ClassiferNN

from baselines.baseline3b import NN_Training
trainer = NN_Training.__new__(NN_Training)  
def eval(ROOT , config_path , checkpoint_path):

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_activity_cls = Group_Activity_Classifer(
        num_classes=config['model']['num_classes']['person_activity']
    )

    model = ClassiferNN(
        person_feature_extraction=person_activity_cls,
        num_classes=config['model']['num_classes']['group_activity']
    )

    model = load_checkpoint_model(
        model=model, 
        checkpoint_path=checkpoint_path, 
        device=device, 
        optimizer=None
    )

    model = model.to(device)

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    test_dataset = Group(
        videos_path=f"{config['data']['videos_path']}",
        annot_path=f"{config['data']['annot_path']}",
        split=config['data']['video_splits']['test'],
        labels=group_activity_labels, 
        transform=test_transforms,
        crops=True,
        seq=False, 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=40,
        collate_fn=trainer.concat,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()

    path = rf"{ROOT}\test\baseline3_eval"
    prefix = "Group Activity Baseline 3 eval on testset"

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
    MODEL_CONFIG = rf"{ROOT}\configs\Baseline3b.yml"    
    CHECKPOINT_PATH = rf"{ROOT}\test\baseline3_eval\final_model.pth"

    metrics = eval(ROOT, MODEL_CONFIG, CHECKPOINT_PATH)
    print(metrics)
  
"""
==================================================
Group Activity Baseline 3 eval on testset
==================================================
Accuracy : 71.77%
Average Loss: 0.9252
F1 Score (Weighted): 0.7182

Classification Report:
              precision    recall  f1-score   support

       r_set       0.72      0.73      0.73      1728
     r_spike       0.78      0.73      0.75      1557
      r-pass       0.67      0.65      0.66      1890
  r_winpoint       0.51      0.62      0.56       783
  l_winpoint       0.57      0.52      0.54       918
      l-pass       0.82      0.74      0.78      2034
     l-spike       0.79      0.81      0.80      1611
       l_set       0.72      0.81      0.76      1512

    accuracy                           0.72     12033
   macro avg       0.70      0.70      0.70     12033
weighted avg       0.72      0.72      0.72     12033
"""