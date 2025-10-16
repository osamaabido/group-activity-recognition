

import os
import yaml
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import Two_stage_Hierarchical
from dataloader.DataLoader import HierarchicalDataLoader, activities_labels
from eval_utils import get_f1_score, plot_confusion_matrix
from helper_utils import load_config , save_checkpoint_model , load_checkpoint_model , setup_logging

def setup_ddp(rank, world_size, master_addr="localhost", master_port="12355"):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

class Baseline9Trainer:
    def __init__(self, config_file_path, project_root, local_rank, world_size):
        self.local_rank = local_rank
        self.world_size = world_size
        self.Project_root = project_root

        # load config
        self.config = load_config(config_file_path)

        # DDP setup (set environment + init process group)
        setup_ddp(self.local_rank, self.world_size)

        # device
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        # set seed (different per process)
        self.set_seed(self.config['experiment']['seed'] + self.local_rank)

        # model
        self.model = Two_stage_Hierarchical(
            person_num_classes=self.config['model']['num_classes']['person_activity'],
            group_num_classes=self.config['model']['num_classes']['group_activity'],
            hidden_size=self.config['model']['hyper_param']['hidden_size'],
            num_layers=self.config['model']['hyper_param']['num_layers']
        ).to(self.device)

        # prepare experiment dir (only rank 0 creates dir and logger)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(
            f"{self.Project_root}/training/baseline9/{self.config['experiment']['output_dir']}",
            f"{self.config['experiment']['name']}_V{self.config['experiment']['version']}_{timestamp}"
        )
        if self.local_rank == 0:
            os.makedirs(self.exp_dir, exist_ok=True)
            self.logger = setup_logging(self.exp_dir)
            config_save_path = os.path.join(self.exp_dir, 'config.yaml')
            with open(config_save_path, 'w') as config_file:
                yaml.dump(self.config, config_file)
            self.logger.info(f"Configuration saved to {config_save_path}")
        else:
            self.logger = None

        # data loaders (prepare_data will also print counts using logger if rank 0)
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self.prepare_data()

        # compute class weights for group-level loss (used in CrossEntropyLoss)
        self.class_weights = self.calculate_class_weights_from_dataset(self.train_loader.dataset)
        self.group_criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        # person criterion with label smoothing
        self.person_criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        # optimizer, scheduler, scaler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=(self.local_rank==0)
        )
        self.scaler = GradScaler()

        # TensorBoard writer (only rank 0)
        if self.local_rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        else:
            self.writer = None

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def concat(self, batch):
        # batch: list of tuples (clip, person_labels, group_labels)
        clips, person_labels, group_labels = zip(*batch)

        max_bboxes = 12
        padded_clips = []
        padded_person_labels = []

        for clip, label in zip(clips, person_labels):
            # clip: [num_bboxes, channels, frames?, H, W] or similar
            num_bboxes = clip.size(0)
            if num_bboxes < max_bboxes:
                pad_shape = (max_bboxes - num_bboxes, *clip.shape[1:])
                clip_padding = torch.zeros(pad_shape, dtype=clip.dtype)
                label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)), dtype=label.dtype)

                clip = torch.cat((clip, clip_padding), dim=0)
                label = torch.cat((label, label_padding), dim=0)

            padded_clips.append(clip)
            padded_person_labels.append(label)

        padded_clips = torch.stack(padded_clips)  # [B, max_bboxes, C, ...]
        padded_person_labels = torch.stack(padded_person_labels)  # [B, max_bboxes, seq_len, num_person_classes?]
        group_labels = torch.stack(group_labels)  # [B, seq_len, num_group_classes?]

        # keep last frame labels as used in original code
        group_labels = group_labels[:, -1, :]  # [B, num_group_classes]
        padded_person_labels = padded_person_labels[:, :, -1, :]  # [B, max_bboxes, num_person_classes]

        return padded_clips, padded_person_labels, group_labels

    def calculate_class_weights_from_dataset(self, dataset):
        """
        Calculate class weights for group-level labels.
        Returns a tensor of size [num_group_classes].
        """
        counts = None
        for i in range(len(dataset)):
            _, _, group_label = dataset[i]  # group_label shape: [seq_len, num_group_classes]
            label_idx = group_label[-1].argmax().item()
            if counts is None:
                num_classes = group_label.size(-1)
                counts = torch.zeros(num_classes, dtype=torch.long)
            counts[label_idx] += 1

        # avoid division by zero
        counts = counts.float()
        counts[counts == 0] = 1.0
        weights = 1.0 / counts
        weights = weights / weights.sum()  # normalized (optional)
        return weights

    def prepare_data(self):
        train_transforms = A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise()
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        train_dataset = HierarchicalDataLoader(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['train'],
            labels=activities_labels,
            transform=train_transforms
        )

        val_dataset = HierarchicalDataLoader(
            videos_path=self.config['data']['videos_path'],
            annot_path=self.config['data']['annot_path'],
            split=self.config['data']['video_splits']['validation'],
            labels=activities_labels,
            transform=val_transforms
        )

        if self.local_rank == 0 and self.logger is not None:
            self.logger.info(f"Number of training samples: {len(train_dataset)}")
            self.logger.info(f"Number of validation samples: {len(val_dataset)}")

        # samplers for DDP
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.local_rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.local_rank)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            collate_fn=self.concat,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=4,
            collate_fn=self.concat,
            pin_memory=True
        )

        return train_loader, val_loader, train_sampler, val_sampler

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, person_labels, group_labels in self.val_loader:
                inputs = inputs.to(self.device)
                person_labels = person_labels.to(self.device)
                group_labels = group_labels.to(self.device)

                with autocast():
                    person_outputs, group_outputs = self.model(inputs)  # adapt to model return
                    group_loss = self.group_criterion(group_outputs, group_labels.argmax(dim=1))
                    person_loss = self.person_criterion(
                        person_outputs.view(-1, person_outputs.size(-1)),
                        person_labels.view(-1, person_labels.size(-1))
                    )
                    loss = group_loss + person_loss

                batch_size = inputs.size(0)
                total_loss += loss.detach() * batch_size

                _, predicted = torch.max(group_outputs.data, 1)
                total += batch_size
                correct += (predicted == group_labels.argmax(dim=1)).sum().item()

                y_true.extend(group_labels.argmax(dim=1).cpu().numpy().tolist())
                y_pred.extend(predicted.cpu().numpy().tolist())

        # aggregate across processes
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

            total_loss = loss_tensor.item()
            total = total_tensor.item()
            correct = correct_tensor.item()

        avg_loss = total_loss / (len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else 1)
        accuracy = 100. * correct / total if total > 0 else 0.0

        if self.local_rank == 0:
            f1_score = get_f1_score(y_true, y_pred, average="weighted")
            if self.writer:
                self.writer.add_scalar('Validation/F1Score', f1_score, epoch)
                fig = plot_confusion_matrix(y_true, y_pred, self.config['model']['num_classes_label']['group_activity'])
                self.writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
                self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
                self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
            return avg_loss, accuracy, f1_score

        return avg_loss, accuracy, 0.0

    def train(self, checkpoint_path=None):
        # wrap model with DDP after it's on correct device
        self.model = DDP(self.model, device_ids=[self.local_rank])

        start_epoch = 0
        best_val_acc = -1.0

        # optionally resume
        if checkpoint_path:
            device = self.device
            model_state, optim_state, loaded_config, exp_dir_loaded, start_epoch = load_checkpoint_model(
                checkpoint_path, self.model, self.optimizer, device
            )
            if self.local_rank == 0 and self.logger:
                self.logger.info(f"Resumed training from epoch {start_epoch}")

        num_epochs = self.config['training']['group_activity']['epochs']
        for epoch in range(start_epoch, num_epochs):
            # set epoch for DistributedSampler
            if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            total = 0
            correct = 0

            for batch_idx, (inputs, person_labels, group_labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                person_labels = person_labels.to(self.device)
                group_labels = group_labels.to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    person_outputs, group_outputs = self.model(inputs)
                    # ensure shapes; adapt if your model returns dicts
                    loss_1 = self.person_criterion(
                        person_outputs.view(-1, person_outputs.size(-1)),
                        person_labels.view(-1, person_labels.size(-1))
                    )
                    loss_2 = self.group_criterion(group_outputs, group_labels.argmax(dim=1))
                    loss = loss_2 + 0.60 * loss_1

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # aggregate loss & accuracy across processes
                loss_tensor = loss.detach()
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor = loss_tensor / dist.get_world_size()
                epoch_loss += loss_tensor.item() * inputs.size(0)

                preds = group_outputs.argmax(dim=1)
                targets = group_labels.argmax(dim=1)
                batch_correct = (preds == targets).sum().item()
                batch_total = targets.size(0)

                # reduce correct/total across processes periodically or at epoch end
                correct += batch_correct
                total += batch_total

                if batch_idx % 100 == 0 and self.local_rank == 0 and self.logger:
                    avg_loss_so_far = epoch_loss / (batch_idx + 1e-9)
                    # compute approx accuracy across this process only (for logging)
                    acc_so_far = 100.0 * correct / (total if total > 0 else 1)
                    step = epoch * len(self.train_loader) + batch_idx
                    if self.writer:
                        self.writer.add_scalar('Training/BatchLoss', avg_loss_so_far, step)
                        self.writer.add_scalar('Training/BatchAccuracy', acc_so_far, step)
                    self.logger.info(f'Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss {avg_loss_so_far:.4f} | Acc {acc_so_far:.2f}%')

            # end epoch: reduce correct/total across processes
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            total_correct = correct_tensor.item()
            total_samples = total_tensor.item()

            # reduce epoch_loss across processes
            epoch_loss_tensor = torch.tensor(epoch_loss, device=self.device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / (self.world_size if self.world_size > 0 else 1)

            epoch_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

            if self.local_rank == 0:
                if self.writer:
                    self.writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
                    self.writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)

            # validation
            val_loss, val_acc, val_f1 = self.validate(epoch)

            if self.local_rank == 0 and self.logger:
                self.logger.info(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
                self.logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

                # scheduler step
                self.scheduler.step(val_loss)

                # save best
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint_model(self.model.module if hasattr(self.model, 'module') else self.model,
                                    self.optimizer, epoch, val_acc, self.config, self.exp_dir, is_best=True)

                # save latest
                save_checkpoint_model(self.model.module if hasattr(self.model, 'module') else self.model,
                                self.optimizer, epoch, val_acc, self.config, self.exp_dir, is_best=False)

                # log lr
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.writer:
                    self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
                self.logger.info(f'Current learning rate: {current_lr}')

        # cleanup
        if self.writer is not None:
            self.writer.close()

        if self.local_rank == 0 and self.logger:
            self.logger.info("Training completed.")

        cleanup_ddp()


def _spawn_worker(local_rank, config_path, project_root, world_size):
    trainer = Baseline9Trainer(config_path, project_root, local_rank, world_size)
    trainer.train(checkpoint_path=None)



def main():
    config_path = r"/kaggle/working/group-activity-recognition/configs/Baseline9.yml"  # set this
    project_root = r"/kaggle/working/group-activity-recognition",
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs found for DDP. Found 0 GPUs.")

    mp.spawn(
        _spawn_worker,
        args=(config_path, project_root, world_size),
        nprocs=world_size,
        join=True
)


if __name__ == "__main__":
    main()
