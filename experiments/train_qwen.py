"""Qwen 2.5 Fine-tuning Pipeline for LVLM.

Implements specialized training for Qwen with:
- LoRA parameter-efficient fine-tuning
- LVLM temporal binding integration
- Gradient accumulation and mixed precision
- Multi-task learning (QA + temporal grounding)

Usage:
    python train_qwen.py --config ../configs/qwen_finetuning.yaml --dataset tvqa --epochs 10
    python train_qwen.py --config ../configs/qwen_finetuning.yaml --dataset tvqa --resume checkpoint.pt
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from parent directory modules
from utils.logging import setup_logger
from utils.config import load_config
from utils.metrics import Metrics, MetricsTracker
from models import LVLM
from models.qwen_adapter import QwenAdapter, QwenLVLMFusion
from data_utils import create_data_loaders

logger = logging.getLogger(__name__)


class QwenTrainer:
    """Specialized trainer for Qwen 2.5 fine-tuning with LVLM."""
    
    def __init__(
        self,
        lvlm_model: nn.Module,
        qwen_adapter: QwenAdapter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        checkpoint_dir: Path,
        freeze_lvlm: bool = True,
    ):
        """Initialize Qwen trainer.
        
        Args:
            lvlm_model: Frozen LVLM instance for feature extraction
            qwen_adapter: Qwen 2.5 adapter with LoRA
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Computation device
            checkpoint_dir: Directory for saving checkpoints
            freeze_lvlm: Whether to freeze LVLM features (Stage 1) or train (Stage 2)
        """
        self.lvlm = lvlm_model
        self.qwen = qwen_adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.num_epochs = config.get('training', {}).get('num_epochs', 10)
        self.lr = config.get('training', {}).get('learning_rate', 2e-4)
        self.batch_size = config.get('training', {}).get('batch_size', 16)
        self.gradient_accumulation_steps = config.get(
            'training', {}
        ).get('gradient_accumulation_steps', 2)
        self.warmup_steps = config.get('training', {}).get('warmup_steps', 500)
        self.max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
        self.use_fp16 = config.get('training', {}).get('use_fp16', True)
        self.freeze_lvlm = freeze_lvlm
        
        # Loss weights for multi-task learning
        loss_config = config.get('loss_weights', {})
        self.loss_weights = {
            'qa': loss_config.get('qa', 1.0),
            'grounding': loss_config.get('grounding', 0.5),
            'contrastive': loss_config.get('contrastive', 0.1),
        }
        
        # Freeze LVLM if Stage 1
        if self.freeze_lvlm:
            for param in self.lvlm.parameters():
                param.requires_grad = False
            logger.info("✅ Stage 1: LVLM features frozen")
        else:
            logger.info("✅ Stage 2: LVLM features trainable (joint fine-tuning)")
        
        # Enable Qwen training
        self.qwen.enable_training()
        
        # Setup optimizer - only for trainable parameters
        trainable_params = [p for p in self.qwen.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=config.get('training', {}).get('weight_decay', 1e-5),
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = self._create_scheduler(total_steps)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience = config.get('training', {}).get('early_stopping_patience', 5)
        self.patience_counter = 0
        
        # WandB setup
        self.use_wandb = HAS_WANDB and config.get('logging', {}).get('use_wandb', True)
        if self.use_wandb:
            self._init_wandb()
        
        logger.info(f"QwenTrainer initialized | LR: {self.lr} | Epochs: {self.num_epochs}")
    
    def _create_scheduler(self, total_steps: int):
        """Create cosine learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(
                max(1, total_steps - self.warmup_steps)
            ))
        
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.get('logging', {}).get('wandb_project', 'lvlm-qwen'),
            name=self.config.get('logging', {}).get('experiment_name', 'qwen-finetuning'),
            config=self.config,
            dir=str(self.checkpoint_dir),
        )
        logger.info("✅ Initialized WandB logging")
    
    def _compute_loss(
        self,
        qwen_output: Dict,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss for Qwen.
        
        Args:
            qwen_output: Output from Qwen adapter
            batch: Batch data containing answers, grounding targets
            
        Returns:
            Total loss and loss dictionary
        """
        losses = {}
        
        # QA Loss: Cross-entropy on answer prediction
        if 'logits' in qwen_output and 'answers' in batch:
            qa_loss = nn.CrossEntropyLoss()(
                qwen_output['logits'],
                batch['answers'],
            )
            losses['qa'] = qa_loss * self.loss_weights['qa']
        
        # Temporal Grounding Loss: Smooth L1 for timestamp prediction
        if qwen_output.get('grounding') is not None and 'grounding_targets' in batch:
            grounding_loss = nn.SmoothL1Loss()(
                qwen_output['grounding'],
                batch['grounding_targets'],
            )
            losses['grounding'] = grounding_loss * self.loss_weights['grounding']
        
        # Total loss
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)
        
        # Convert to float dict for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        loss_dict['total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.lvlm.eval() if self.freeze_lvlm else self.lvlm.train()
        self.qwen.train()
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
            unit="batch",
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Extract features from LVLM (temporal binding)
            with torch.no_grad() if self.freeze_lvlm else torch.enable_grad():
                lvlm_output = self.lvlm(batch)
                temporal_features = lvlm_output.get('temporal_features')
                memory_nodes = lvlm_output.get('memory_nodes')
            
            # Forward pass through Qwen
            with autocast(enabled=self.use_fp16):
                qwen_output = self.qwen(
                    question_ids=batch['question_ids'],
                    temporal_features=temporal_features,
                    memory_nodes=memory_nodes,
                    return_grounding=True,
                )
                
                loss, loss_dict = self._compute_loss(qwen_output, batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.qwen.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track losses
            for key, val in loss_dict.items():
                epoch_losses[key] += val
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss_dict['total'],
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Log to WandB
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    f"train/{k}": v for k, v in loss_dict.items()
                }, step=self.global_step)
        
        # Average losses
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_losses.get('total', 0):.4f}")
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Run validation epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.lvlm.eval()
        self.qwen.eval()
        
        metrics_tracker = MetricsTracker()
        val_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # LVLM forward
                lvlm_output = self.lvlm(batch)
                temporal_features = lvlm_output.get('temporal_features')
                memory_nodes = lvlm_output.get('memory_nodes')
                
                # Qwen forward
                qwen_output = self.qwen(
                    question_ids=batch['question_ids'],
                    temporal_features=temporal_features,
                    memory_nodes=memory_nodes,
                )
                
                loss, loss_dict = self._compute_loss(qwen_output, batch)
                
                # Track losses
                for key, val in loss_dict.items():
                    val_losses[key] += val
                
                # Compute metrics
                if 'logits' in qwen_output and 'answers' in batch:
                    preds = qwen_output['logits'].argmax(dim=-1)
                    targets = batch['answers']
                    accuracy = (preds == targets).float().mean().item()
                    metrics_tracker.metrics.track_accuracy(accuracy)
                
                num_batches += 1
                progress_bar.set_postfix({'val_loss': loss_dict['total']})
        
        # Average metrics
        avg_val_losses = {k: v / max(1, num_batches) for k, v in val_losses.items()}
        avg_accuracy = metrics_tracker.metrics.get_accuracy() if hasattr(metrics_tracker, 'metrics') else 0.0
        
        logger.info(f"Validation - Loss: {avg_val_losses.get('total', 0):.4f} | Accuracy: {avg_accuracy:.4f}")
        
        return {
            **avg_val_losses,
            'accuracy': avg_accuracy,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'config': self.config,
        }
        
        # Save LoRA weights
        if is_best:
            save_path = self.checkpoint_dir / "best_qwen_lora"
        else:
            save_path = self.checkpoint_dir / f"qwen_lora_epoch{epoch}"
        
        self.qwen.save_lora_weights(str(save_path))
        logger.info(f"✅ Checkpoint saved: {save_path}")
    
    def fit(self):
        """Run complete training loop."""
        logger.info("=" * 80)
        logger.info("STARTING QWEN FINE-TUNING")
        logger.info("=" * 80)
        
        for epoch in range(self.num_epochs):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Early stopping
            current_val_loss = val_metrics.get('total', float('inf'))
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    **{f"train/{k}": v for k, v in train_losses.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }, step=self.global_step)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 for video QA")
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--dataset', type=str, default='tvqa', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=None, help='Override num epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--freeze_lvlm', action='store_true', default=True,
                       help='Freeze LVLM (Stage 1) or train jointly (Stage 2)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_qwen',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    setup_logger(logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    logger.info(f"Config loaded: {args.config}")
    
    # Create models
    logger.info("Loading LVLM model...")
    lvlm_model = LVLM(
        **config.get('model', {})
    ).to(device)
    
    logger.info("Loading Qwen 2.5 adapter...")
    qwen_adapter = QwenAdapter(
        model_name=config.get('qwen', {}).get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
        lora_rank=config.get('qwen', {}).get('lora_rank', 16),
        use_4bit=config.get('qwen', {}).get('use_4bit', True),
        device=str(device),
    )
    
    # Create data loaders
    logger.info(f"Creating {args.dataset} data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset=args.dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
    )
    
    # Create trainer
    trainer = QwenTrainer(
        lvlm_model=lvlm_model,
        qwen_adapter=qwen_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=Path(args.output_dir),
        freeze_lvlm=args.freeze_lvlm,
    )
    
    # Train
    trainer.fit()
    
    logger.info("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
