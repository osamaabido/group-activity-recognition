from Trainer import Trainer
from Trainer import concat_group
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.LstmGroup import LSTMGroup
from dataloader.DataLoader import Group, group_activity_labels
from helper_utils import load_config
from configs.Paths import File_Paths

Project_Root = File_Paths.Project_Root
Config_file_path = File_Paths.Config_file_path_Baseline4

def main():
    # Load configuration from file
    config_path = File_Paths.Config_file_path_Baseline4
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
            crops=False,
            seq=True,
            labels=group_activity_labels, 
            transform=train_transforms
        )

    val_dataset = Group(
            videos_path=config['data']['videos_path'],
            annot_path=config['data']['annot_path'],
            split=config['data']['video_splits']['validation'],
            crops=False,
            seq=True,
            labels=group_activity_labels,
            transform=val_transforms
        )
    
    train_loader = DataLoader(
            train_dataset,
            collate_fn = Trainer.concat_group , 
            batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
            )
    val_loader = DataLoader(
            val_dataset,
            collate_fn = Trainer.concat_group ,
            batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model  = LSTMGroup(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_classes=config['model']['num_classes'],
        )
    # Initialize and run the trainer
    trainer = Trainer(
        config_file_path=Config_file_path, 
        project_root=Project_Root , 
        model = model
    )
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
