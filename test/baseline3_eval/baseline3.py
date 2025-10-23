import os 
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)


import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from helper_utils import load_config , load_checkpoint_model
from eval_utils import model_eval
from dataloader import Group, group_activity_labels
from models import GroupActivityClassifer , PeronActivityClassifer

from baselines.baseline3b import NN_Training
trainer = NN_Training.__new__(NN_Training)  
def eval(ROOT , config_path , checkpoint_path):

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_activity_cls = PeronActivityClassifer(
        num_classes=config.model['num_classes']['person_activity']
    )

    model = GroupActivityClassifer(
        person_feature_extraction=person_activity_cls,
        num_classes=config.model['num_classes']['group_activity']
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
    MODEL_CONFIG = rf"{ROOT}\configs\Baseline3.yml"    
    CHECKPOINT_PATH = rf"{ROOT}\test\baseline3_eval\final_model.pth"

    metrics = eval(ROOT, MODEL_CONFIG, CHECKPOINT_PATH)
    print(metrics)
  
