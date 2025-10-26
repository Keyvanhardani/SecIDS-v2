"""
PyTorch Lightning Trainers for IDS Models
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
from typing import Dict, Any, Optional
import sys
sys.path.append('..')
from models import BaseIDS, MultiTaskIDS
from models.multitask import MultiTaskLoss


class IDSTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for single-task IDS training.
    """

    def __init__(
        self,
        model: BaseIDS,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        max_epochs: int = 100,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: IDS model (TCN, Mamba, etc.)
            learning_rate: Initial LR
            weight_decay: L2 regularization
            scheduler: "cosine" or "plateau"
            max_epochs: For cosine annealing
            class_weights: For imbalanced datasets
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.max_epochs = max_epochs

        # Loss
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        features = batch['features']  # [batch, seq_len, num_features]
        labels = batch['label']       # [batch]

        # Forward
        logits = self.model(features)

        # Loss
        loss = self.criterion(logits, labels)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        features = batch['features']
        labels = batch['label']

        # Forward
        logits = self.model(features)
        loss = self.criterion(logits, labels)

        # Predictions
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of attack class

        # Metrics
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.val_auroc(probs, labels)

        # Logging
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_epoch=True)
        self.log('val/recall', self.val_recall, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True, prog_bar=True)
        self.log('val/auroc', self.val_auroc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }

        elif self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch'
                }
            }

        else:
            return optimizer


class MultiTaskIDSTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for multi-task IDS training.
    """

    def __init__(
        self,
        model: MultiTaskIDS,
        task_names: list,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        max_epochs: int = 100
    ):
        super().__init__()
        self.model = model
        self.task_names = task_names
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.max_epochs = max_epochs

        # Multi-task loss with uncertainty weighting
        self.criterion = MultiTaskLoss(task_names)

        # Per-task metrics
        self.train_metrics = nn.ModuleDict({
            task: nn.ModuleDict({
                'acc': Accuracy(task="binary"),
                'f1': F1Score(task="binary")
            })
            for task in task_names
        })

        self.val_metrics = nn.ModuleDict({
            task: nn.ModuleDict({
                'acc': Accuracy(task="binary"),
                'precision': Precision(task="binary"),
                'recall': Recall(task="binary"),
                'f1': F1Score(task="binary"),
                'auroc': AUROC(task="binary")
            })
            for task in task_names
        })

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for multi-task"""
        features = batch['features']
        labels = batch['labels']  # Dict[task_name -> labels]

        # Forward
        outputs = self.model(features)  # Dict[task_name -> logits]

        # Multi-task loss
        loss_dict = self.criterion(outputs, labels)
        total_loss = loss_dict['total']

        # Metrics per task
        for task in self.task_names:
            if task in outputs and task in labels:
                preds = torch.argmax(outputs[task], dim=1)
                self.train_metrics[task]['acc'](preds, labels[task])
                self.train_metrics[task]['f1'](preds, labels[task])

        # Logging
        self.log('train/loss_total', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log task-specific losses and weights
        for task in self.task_names:
            if task in loss_dict['task_losses']:
                self.log(f'train/loss_{task}', loss_dict['task_losses'][task], on_epoch=True)
                self.log(f'train/weight_{task}', loss_dict['weights'][task], on_epoch=True)
                self.log(f'train/acc_{task}', self.train_metrics[task]['acc'], on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step for multi-task"""
        features = batch['features']
        labels = batch['labels']

        # Forward
        outputs = self.model(features)

        # Loss
        loss_dict = self.criterion(outputs, labels)
        total_loss = loss_dict['total']

        # Metrics per task
        for task in self.task_names:
            if task in outputs and task in labels:
                preds = torch.argmax(outputs[task], dim=1)
                probs = torch.softmax(outputs[task], dim=1)[:, 1]

                self.val_metrics[task]['acc'](preds, labels[task])
                self.val_metrics[task]['precision'](preds, labels[task])
                self.val_metrics[task]['recall'](preds, labels[task])
                self.val_metrics[task]['f1'](preds, labels[task])
                self.val_metrics[task]['auroc'](probs, labels[task])

        # Logging
        self.log('val/loss_total', total_loss, on_epoch=True, prog_bar=True)

        for task in self.task_names:
            if task in outputs and task in labels:
                self.log(f'val/acc_{task}', self.val_metrics[task]['acc'], on_epoch=True)
                self.log(f'val/precision_{task}', self.val_metrics[task]['precision'], on_epoch=True)
                self.log(f'val/recall_{task}', self.val_metrics[task]['recall'], on_epoch=True)
                self.log(f'val/f1_{task}', self.val_metrics[task]['f1'], on_epoch=True, prog_bar=True)
                self.log(f'val/auroc_{task}', self.val_metrics[task]['auroc'], on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }

        elif self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss_total',
                    'interval': 'epoch'
                }
            }

        return optimizer
