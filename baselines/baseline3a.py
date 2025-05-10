import os
import albumentations as A
import torch
import random
import numpy as np
import yaml
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from models.GroupActivityClassifer import Group_Activity_Classifer
from dataloader.DataLoader import Group, group_activity_labels , Person , person_activity_labels
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint_model
from CVR16.baselines.trainer import Tranier
class Person_Tranier(Tranier):
    def __init__(self, config_file_path, project_root):
        self.Project_Root = project_root
        self.config = load_config(config_file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(
            f"{self.Project_Root}/training/baseline3/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        self.set_seed(self.config['experiment']['seed'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Group_Activity_Classifer(num_classes=len(person_activity_labels)).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr= self.config['training']['learning_rate'],
                        weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True ) 
        self.scaler = GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = setup_logging(self.exp_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        self.train_loader, self.val_loader = self.prepare_data()
        self.class_names = self.config['model']['num_clases_label']
        config_save_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_save_path, 'w') as config_file:
            yaml.dump(self.config, config_file)
        self.logger.info(f"Configuration saved to {config_save_path}")
        super(Tranier).__init__()
    def prepare_data(self):
        new_train_transforms = A.Compose([
            A.Resize(256, 256),  
            A.RandomRotate90(),  
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        new_val_transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        new_train_dataset = Person(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['train'],
            labels=person_activity_labels,
            transform=new_train_transforms
        )

        new_val_dataset = Person(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['validation'],
            labels=person_activity_labels,
            transform=new_val_transforms
        )

        self.logger.info(f"New training dataset size: {len(new_train_dataset)}")
        self.logger.info(f"New validation dataset size: {len(new_val_dataset)}")

        new_train_loader = DataLoader(
            new_train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True
        )

        new_val_loader = DataLoader(
            new_val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
        )

        return new_train_loader, new_val_loader