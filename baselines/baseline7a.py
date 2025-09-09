import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
import random
import albumentations as A
import torch.multiprocessing as mp
from  datetime import datetime 
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from models import Person_Activity_Temporal

from dataloader import Person , person_activity_labels
from eval_utils import get_f1_score , plot_confusion_matrix
from helper_utils import load_config , save_checkpoint_model , load_checkpoint_model , setup_logging


class BaseLine_7A_Trainer():
    def __init__(self , config_file_path , Project_root):
        self.config = load_config(config_file_path)
        self.Project_root = Project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir  = os.path.join(
            f"{self.Project_root} / training /baseline7a / {self.config['experiment']['output_dir']}" , 
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
        os.makedirs(self.exp_dir , exist_ok=True)
        self.set_seed(self.config['experiment']['seed'])

        self.model = Person_Activity_Temporal(
            num_classes= self.config['model']['num_classes']['person_activity'],
            hidden_size= self.config['model']['hyper_param']['person_activity']['hidden_size'] , 
            num_layers=self.config['model']['hyper_param']['person_activity']['num_layers']
            ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters() , 
            lr=self.config['training']['person_activity']['learning_rate'] , 
            weight_decay=self.config['training']['person_activity']['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1,
            patience=0.3, 
            verbose=True
        )
        self.scaler = GradScaler()

        self.criterion = nn.CrossEntropyLoss()
        self.logger = setup_logging(self.exp_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))

        self.train_loader, self.val_loader = self.prepare_data()
        self.class_names = self.config['model']['num_classes_label']['person_activity']
        config_save_file = os.path.join(self.exp_dir , 'config.yaml')
        with open(config_save_file , 'w') as f:
            yaml.dump(self.config , f)
        self.logger.info(f"Configuration saved to {config_save_file}")


    def concat(self , batch):
        clips, labels = zip(*batch)  
    
        max_bboxes = 12  
        padded_clips = []
        padded_labels = []

        for clip, label in zip(clips, labels) :
            num_bboxes = clip.size(0)
            if num_bboxes < max_bboxes:
                clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
                label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
                
                clip = torch.cat((clip, clip_padding), dim=0)
                label = torch.cat((label, label_padding), dim=0)
                
            padded_clips.append(clip)
            padded_labels.append(label)
        
        padded_clips = torch.stack(padded_clips)
        padded_labels = torch.stack(padded_labels)
        
        padded_labels = padded_labels[:, :, -1, :]  # utils the label of last frame for each player
        b, bb, num_class = padded_labels.shape # batch, bbox, num_clases
        padded_labels = padded_labels.view(b*bb, num_class)

        return padded_clips, padded_labels


    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
    
    def prepare_data(self):
        train_transforms = A.Compose([
            A.Resize(height=224, width=224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3,7)),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.RandomBrightnessContrast(),
                A.GaussNoise()

            ] , p=0.8),
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()], p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        train_dataset = Person(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['train'],
            labels=person_activity_labels,
            seq=True,
            transform=train_transforms
           
        )

        val_dataset = Person(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['validation'],
            labels=person_activity_labels,
            seq=True,
            transform=val_transforms

        )
        
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['person_activity']['batch_size'],
            collate_fn=self.concat,
            shuffle=True, num_workers=4, pin_memory=True
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['person_activity']['batch_size'],
            collate_fn=self.concat,
            shuffle=False, num_workers=4, pin_memory=True
            )
        return train_loader, val_loader
    
    def validate(self , epoch):
        self.model.eval()
        total_loss  , correct  , total = 0.0 , 0.0 , 0.0
        y_true , y_pred = [] , []

        with torch.no_grad():
            for inputs , targets in self.val_loader:
                inputs , targets  =  inputs.to(self.device) , targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                predicted = outputs.argmax(1)
                target_class = targets.argmax(1)
                total += targets.size(0)
                correct += predicted.eq(target_class).sum().item()

                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        f1_score = get_f1_score(y_true, y_pred, average="weighted")
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Validation/F1Score', f1_score, epoch)
        self.writer.add_figure('Validation/ConfusionMatrix', plot_confusion_matrix( y_true, y_pred, class_names = self.config["model"]['num_classes_label']['person_activity'] , save_path = "/kaggle/working/"))

        self.logger.info(f"Epoch {epoch} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f}")
        return avg_loss, accuracy
    
    def train(self , checkpoint_path=None):
        if checkpoint_path:
            self.model, self.optimizer, self.loaded_config, self.exp_dir, self.start_epoch = load_checkpoint_model(checkpoint_path, self.model, self.optimizer, self.device)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exp_dir = os.path.join(
            f"{self.Project_Root}/training/baseline7a/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
            os.makedirs(self.exp_dir, exist_ok=True)
            self.logger = setup_logging(self.exp_dir)
            
            if(self.loaded_config):
                # config = loaded_config
                self.logger.info(f"Resumed training from epoch {self.start_epoch}")

        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exp_dir = os.path.join(
            f"{self.Project_root}/training/baseline7a/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
            )
            os.makedirs(self.exp_dir, exist_ok=True)
            self.logger = setup_logging(self.exp_dir)
            
            
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['person_activity']['epochs']):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0  
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['person_activity']['epochs']}")

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with  autocast("cuda", dtype=torch.float16):
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                predicted_classes = preds.argmax(dim=1)
                true_classes = labels.argmax(dim=1)
                total_samples += labels.size(0)
                total_correct += (predicted_classes == true_classes).sum().item()
                if batch_idx % 100 == 0:
                    acc = total_correct / total_samples
                    self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f} - Accuracy: {acc:.4f}")

            avg_loss = total_loss / len(self.train_loader)
            avg_accuracy = 100. * total_correct / total_samples
            self.writer.add_scalar('loss/train', avg_loss, epoch)
            self.writer.add_scalar('accuracy/train', avg_accuracy, epoch)

            self.logger.info(f"Epoch {epoch + 1} Summary: Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.2f}%")

            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            self.logger.info(f"Current learning rate: {current_lr}")

            save_checkpoint_model(self.model, self.optimizer, epoch, val_acc,  self.exp_dir , self.config)

        self.writer.close()
        self.save_model()

    def save_model(self):
        final_model_path = os.path.join(self.exp_dir, 'final_model.pth')
        torch.save({
            'epochs': self.config['training']['epochs'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, final_model_path)
        self.logger.info(f"Training completed. Final model saved to: {final_model_path}")