import sys
import os 
import albumentations as A
import torch 
import random
import numpy as np
import yaml
import torch.nn as nn
import torch.optim as optim
from datetime import  datetime 
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast , GradScaler
from torch.utils.data import DataLoader
from models.PeronActivityClassifer import ClassiferNN
from models.GroupActivityClassifer import Group_Activity_Classifer
from torch.utils.tensorboard.writer import SummaryWriter
from dataloader.DataLoader import Group, group_activity_labels , Person , person_activity_labels
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint_model , load_checkpoint_model
from baselines.trainer import Tranier
Project_Root = r"H:\Group-Activity-Recognition"

sys.path.append(Project_Root)

class NN_Training(Tranier):
    def __init__(self , config_file_path , project_root , person_activity_checkpoints):
        self.Project_Root = project_root
        self.config = load_config(config_file_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modela = Group_Activity_Classifer(num_classes=self.config['model']['num_classes']['person_activity'])
        self.person_activity = load_checkpoint_model(
            checkpoint_path=person_activity_checkpoints,
            model = self.modela ,
            device=self.device, 
            optimizer=None
            )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(
            f"{self.Project_Root}/training/baseline3/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        self.set_seed(self.config['experiment']['seed'])
        self.modelb = ClassiferNN(
            person_feature_extraction=self.person_activity, 
        num_classes=self.config.model['num_classes']['group_activity']
        )
        self.optimizer = torch.optim.AdamW(self.modelb.parameters(),
                        lr= self.config['training']['learning_rate'],
                        weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True ) 
        self.scaler = GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = setup_logging(self.exp_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        self.train_loader, self.val_loader = self.prepare_data()
        self.class_names = self.config['model']['num_classes_label']["group_activity"]
        config_save_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_save_path, 'w') as config_file:
            yaml.dump(self.config, config_file)
        self.logger.info(f"Configuration saved to {config_save_path}")
        super(Tranier , self).__init__()
        
    def concat(self , batch):
            clips , labels =  zip(*batch)
            max_box = 12
            single_clips = []
            single_labels = []
            
            for clip , label in zip(clips , labels):
                num_boxes = clip.size(0)
                if num_boxes < max_box:
                    clip_padding = torch.zeros((max_box - num_boxes, clip.size(1), clip.size(2), clip.size(3)))
                    clip = torch.cat((clip, clip_padding), dim=0)
                
                single_clips.append(clip)
                single_labels.append(label)
            
            single_clips = torch.stack(single_clips)
            single_labels = torch.stack(single_labels)
            
            return single_clips, single_labels
        
    def prepare_data(self):
        train_transforms = A.Compose([
           A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise()
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.05),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        train_dataset = Group(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['train'],
            labels=group_activity_labels,
            transform=train_transforms,
            crops=True,
            seq=False,
        )

        val_dataset = Group(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['validation'],
            labels=group_activity_labels,
            transform=val_transforms,
            crops=True,
            seq=False, 
        )

        self.logger.info(f"New training dataset size: {len(train_dataset)}")
        self.logger.info(f"New validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn= self.concat,
            shuffle=True, num_workers=4, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn= self.concat,
            shuffle=False, num_workers=4, pin_memory=True
        )

        return train_loader, val_loader
        