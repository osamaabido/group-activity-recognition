import torch
from baseline.Trainer import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.GroupActivityClassifer import Group_Activity_Classifer
from models.PeronActivityClassifer import ClassiferNN
from dataloader.DataLoader import Group, group_activity_labels 
from dataloader.DataLoader import Person , person_activity_labels
from helper_utils import load_config
from configs.Paths import File_Paths
from helper_utils import load_config, load_checkpoint_model  


Project_Root = File_Paths.Project_Root
Config_file_path = File_Paths.Config_file_path_Baseline3b

def main(person_activity_checkpoints):
    # Load configuration from file
    config_path = File_Paths.Config_file_path_Baseline3b

    config = load_config(config_path)
    train_transforms = A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise()
            ], p=0.80),
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()], p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    train_dataset = Group(
            videos_path=config['data']['videos_path'],
            annot_path=config['data']['annot_path'],
            split=config['data']['video_splits']['train'],
            crops=True,
            seq=False,
            labels=group_activity_labels, 
            transform=train_transforms
        )

    val_dataset = Group(
            videos_path=config['data']['videos_path'],
            annot_path=config['data']['annot_path'],
            split=config['data']['video_splits']['validation'],
            crops=True,
            seq=False,
            labels=group_activity_labels,
            transform=val_transforms
        )
    
    train_loader = DataLoader(
            train_dataset,
            collate_fn = Trainer.concat , 
            batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
            )
    val_loader = DataLoader(
            val_dataset,
            collate_fn = Trainer.concat , 
            batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    modela = Group_Activity_Classifer(num_classes=config['model']['num_classes']['person_activity'])
    person_activity = load_checkpoint_model(
            checkpoint_path=person_activity_checkpoints,
            model = modela ,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            optimizer=None
            )
    
    model = ClassiferNN(
            person_feature_extraction=person_activity, 
        num_classes=config['model']['num_classes']['group_activity']
        )

    # Initialize and run the trainer
    trainer = Trainer(
        config_file_path=Config_file_path ,
        project_root= Project_Root,
        model = model
    )
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main(person_activity_checkpoints = r"H:\Group-Activity-Recognition\group-activity-recognition\checkpoint_epoch _ 9.pkl")

