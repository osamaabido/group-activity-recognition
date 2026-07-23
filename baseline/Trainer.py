import os
import sys
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config, setup_logging, save_checkpoint_model, load_checkpoint_model  
from configs.Paths import File_Paths

Project_Root = File_Paths.Project_Root

sys.path.append(Project_Root)


class Trainer:
    def __init__(self, config_file_path, project_root, model, checkpoint_path=None):
        self.Project_Root = project_root
        self.config = load_config(config_file_path)
        self.checkpoint_path = checkpoint_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],   
            weight_decay=self.config['training']['weight_decay']
        )

        if self.checkpoint_path:
            self.model, self.optimizer, self.loaded_config, self.exp_dir, self.start_epoch = load_checkpoint_model(
                self.checkpoint_path, self.model, self.optimizer, self.device
            )
            self.exp_dir = File_Paths.Exp_Dir_Baseline1
            self.logger = setup_logging(self.exp_dir)

            if self.loaded_config:
                self.logger.info(f"Resumed training from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exp_dir = os.path.join(
                f"{self.Project_Root}/training/{self.config['experiment']['baseline']}/{self.config['experiment']['output_dir']}",
                f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
            )
            os.makedirs(self.exp_dir, exist_ok=True)
            self.logger = setup_logging(self.exp_dir)

        self.set_seed(self.config['experiment']['seed'])

        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
            )
            if self.config['training'].get('lr_scheduler', False)
            else None
        )

        self.scaler = GradScaler('cuda')


        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        self.class_names = self.config['model']['num_classes_label']

        config_save_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_save_path, 'w') as config_file:
            yaml.dump(self.config, config_file)
        self.logger.info(f"Configuration saved to {config_save_path}")
    
    
    def concat_group(self , batch):
        clips , label = zip(*batch)
        clips , label  = torch.stack(clips , dim =0) , torch.stack(label , dim =0)
        labels = label[:, -1, :]  
        return clips, labels
    

    def concat(self, batch):
        clips, labels = zip(*batch)
        max_bboxes, padded_clips, padded_labels = 12, [], []

        for clip, label in zip(clips, labels):
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

        padded_labels = padded_labels[:, :, -1, :]  # label of last frame for each player
        b, bb, num_class = padded_labels.shape
        padded_labels = padded_labels.view(b * bb, num_class)

        return padded_clips, padded_labels

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def calculate_class_weights(self, train_loader):
        total_samples = len(train_loader.dataset)  
        labels = [label.argmax().item() for batch in train_loader for label in batch[1]]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights / class_weights.sum()
        self.logger.info(f"Class Weights: {class_weights.tolist()}")
        return class_weights.to(self.device)

    def validate(self, epoch, val_loader):  
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        
                predicted = outputs.argmax(dim=1)
                target_class = targets.argmax(dim=1)

                total += targets.size(0)
                correct += predicted.eq(target_class).sum().item()

                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(val_loader)  
        accuracy = 100. * correct / total
        f1_score = get_f1_score(y_true, y_pred, average="weighted")

        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Validation/F1Score', f1_score, epoch)
        self.writer.add_figure(
            'Validation/ConfusionMatrix',
            plot_confusion_matrix(
                y_true, y_pred,
                class_names=self.config["model"]['num_classes_label'],
                save_path="/kaggle/working/"
            )
        )

        self.logger.info(f"Epoch {epoch} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f}")
        return avg_loss, accuracy

    def train(self, train_loader, val_loader):  
        #class_weights = self.calculate_class_weights(train_loader)
        self.criterion = torch.nn.CrossEntropyLoss(
            #weight=class_weights
        )

        self.logger.info("Starting training...")

        start = self.start_epoch if self.checkpoint_path else 0

        for epoch in range(start, self.config['training']['epochs']):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            for batch_idx, (images, labels) in enumerate(train_loader):
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
                    self.logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Accuracy: {acc:.4f}")

            avg_loss = total_loss / len(train_loader)
            avg_accuracy = 100. * total_correct / total_samples
            self.writer.add_scalar('loss/train', avg_loss, epoch)
            self.writer.add_scalar('accuracy/train', avg_accuracy, epoch)

            self.logger.info(f"Epoch {epoch + 1} Summary: Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.2f}%")

            val_loss, val_acc = self.validate(epoch, val_loader)  

            if self.scheduler:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            self.logger.info(f"Current learning rate: {current_lr}")

            save_checkpoint_model(self.model, self.optimizer, epoch, val_acc, self.exp_dir, self.config)

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