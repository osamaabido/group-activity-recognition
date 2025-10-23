import sys 
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.cuda.amp import autocast, GradScaler

from models.LstmGroup import LSTMGroup
from dataloader.DataLoader import Group, group_activity_labels
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint_model
from baselines.trainer import Tranier

Project_Root = r"/kaggle/working/group-activity-recognition"
sys.path.append(os.path.abspath(Project_Root))

from dataloader.DataLoader import Group, group_activity_labels
from eval_utils import get_f1_score , plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint_model



class LSTM_Trainer():
    def __init__(self , config_file_path, project_root):
        self.Project_Root = project_root
        self.config =  load_config(config_file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(
            f"{self.Project_Root}/training/baseline4/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
        self.logger = setup_logging(self.exp_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.exp_dir, exist_ok=True)
        self.model  = LSTMGroup(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=self.config['model']['num_classes'],
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                        lr= self.config['training']['learning_rate'],
                        weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True )
        self.scaler = GradScaler()
        self.train_loader, self.val_loader = self.prepare_data()
        class_weights = self.calculate_class_weights(self.train_loader)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        self.train_loader, self.val_loader = self.prepare_data()
        self.class_names = self.config['model']['num_classes_label']
        config_save_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_save_path, 'w') as config_file:
            yaml.dump(self.config, config_file)
        self.logger.info(f"Configuration saved to {config_save_path}")
    
    def concat(self , batch):
        clips , label = zip(*batch)
        clips , label  = torch.stack(clips , dim =0) , torch.stack(label , dim =0)
        labels = label[:, -1, :]  
        return clips, labels
        
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
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['train'],
            crops=False,
            seq=True,
            labels=group_activity_labels, 
            transform=train_transforms
        )

        val_dataset = Group(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['validation'],
            crops=False,
            seq=True,
            labels=group_activity_labels,
            transform=val_transforms
        )
        
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.concat,
            shuffle=True, num_workers=4, pin_memory=True
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.concat,
            shuffle=False, num_workers=4, pin_memory=True
            )
        return train_loader, val_loader
    
    def calculate_class_weights(self, train_loader):
        total_samples = len(train_loader)
        labels = [label.argmax().item() for batch in train_loader for label in batch[1]]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights / class_weights.sum()  
        self.logger.info(f"Class Weights: {class_weights.tolist()}")
        
        return class_weights.to(self.device)
    
        
    def validate(self , epoch):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs , targets in self.val_loader:
                inputs , targets  =  inputs.to(self.device) , targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                _, target_class = targets.max(1)
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
        self.writer.add_figure('Validation/ConfusionMatrix', plot_confusion_matrix( y_true, y_pred, class_names = self.config["model"]['num_classes_label'] , save_path = "/kaggle/working/"))

        self.logger.info(f"Epoch {epoch} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f}")
        return avg_loss, accuracy
        
    def train(self):
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0  
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast(dtype=torch.float16):
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
            'epoch': self.config['training']['epochs'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, final_model_path)
        self.logger.info(f"Training completed. Final model saved to: {final_model_path}")
           
if __name__ == "__main__":
    trainer = LSTM_Trainer(
        config_file_path=r"/kaggle/working/group-activity-recognition/configs/Baseline4.yml",
        project_root=r"/kaggle/working/group-activity-recognition",
    )
    trainer.train()