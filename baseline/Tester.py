
import os
import sys
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from datetime import datetime
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from models.GroupActivityClassifer import Group_Activity_Classifer
from dataloader.DataLoader import Group, group_activity_labels
from eval_utils import get_f1_score, plot_confusion_matrix , save_classification_report
from helper_utils import load_config, setup_logging, save_checkpoint_model, load_checkpoint_model  
import tqdm
Project_Root = r"H:\Group-Activity-Recognition"
Config_file_path = r"H:\Group-Activity-Recognition\CVR16\configs\Baseline1.yml"

sys.path.append(Project_Root)

        
    

class Evaluator:
    def __init__(self, config_file_path, model):
        """Initialize the Evaluator with a model and device."""
        self.config = load_config(config_file_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(
                f"{self.Project_Root}/training/{self.config['experiment']['baseline']}/{self.config['experiment']['output_dir']}",
                f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
            )
        os.makedirs(self.exp_dir, exist_ok=True)
        self.logger = setup_logging(self.exp_dir)
        self.class_names = self.config['model']['num_classes_label']
        self.final_model_path = os.path.join(self.exp_dir, 'final_model.pth')
    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        self.model.load_state_dict(torch.load(self.final_model_path, map_location=self.device))
        self.logger.info(f"Loaded model from {self.final_model_path}")

        self.model.eval()
        all_labels = []
        all_preds = []
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                del outputs
                torch.cuda.empty_cache()  # optionally clear cache after heavy operations
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        avg_loss = running_loss / len(dataloader)
        f1_score = get_f1_score(all_labels, all_preds , average='weighted')
        self.logger.info(f"Test Loss: {avg_loss:.4f}, F1 Score: {f1_score:.4f}")

        report_path = os.path.join(self.config.save_dir, 'classification_report.txt')
        conf_matrix_path = os.path.join(self.config.save_dir, 'confusion_matrix.png')
        save_classification_report(all_labels, all_preds, self.class_names, report_path)
        plot_confusion_matrix(all_labels, all_preds, self.class_names, conf_matrix_path, 'Confusion Matrix')

