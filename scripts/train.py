#!/usr/bin/env python3
"""Main training script for multi-modal summarization."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import (
    setup_logging, set_seed, get_device, load_config, 
    save_config, create_directories, EarlyStopping
)
from data import create_data_loaders, collate_fn
from models import MultiModalSummarizer
from eval import MultiModalEvaluator, evaluate_model_predictions
from viz import MultiModalVisualizer


class MultiModalTrainer:
    """Trainer class for multi-modal summarization."""
    
    def __init__(self, config: DictConfig):
        """Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set up logging and device
        setup_logging(config.logging.level, config.logging.log_dir)
        set_seed(config.seed)
        self.device = get_device(config.device.type, config.device.fallback_to_cpu)
        
        # Create directories
        create_directories(config)
        
        # Initialize model
        self.model = MultiModalSummarizer(config).to(self.device)
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )
        
        # Initialize evaluator and visualizer
        self.evaluator = MultiModalEvaluator()
        self.visualizer = MultiModalVisualizer()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=5)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Initialize wandb if enabled
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
                tags=config.logging.wandb.tags,
                config=OmegaConf.to_container(config, resolve=True)
            )
        
        logging.info(f"Initialized trainer with {self.model.__class__.__name__}")
        logging.info(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                images=batch["images"]
            )
            
            # Compute loss (simplified - using text summarization loss)
            # In practice, you'd want a more sophisticated loss function
            loss = self.compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to wandb
            if self.config.logging.use_wandb and batch_idx % self.config.logging.log_every == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    images=batch["images"]
                )
                
                # Compute loss
                loss = self.compute_loss(outputs, batch)
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions for evaluation
                batch_predictions = self.generate_predictions(batch)
                predictions.extend(batch_predictions)
                references.extend(batch["summaries"])
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Compute evaluation metrics
        metrics = evaluate_model_predictions(predictions, references)
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for the batch.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Computed loss
        """
        # Simplified loss computation
        # In practice, you'd want to implement proper multi-modal loss
        
        # Use text summarization loss as proxy
        text_outputs = self.model.text_summarizer(
            batch["text_input_ids"],
            batch["text_attention_mask"]
        )
        
        # Cross-entropy loss for text summarization
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create target labels (simplified)
        target_ids = batch["summary_input_ids"]
        
        # Compute loss
        loss = loss_fn(
            text_outputs.logits.view(-1, text_outputs.logits.size(-1)),
            target_ids.view(-1)
        )
        
        return loss
    
    def generate_predictions(self, batch: Dict[str, Any]) -> list:
        """Generate predictions for evaluation.
        
        Args:
            batch: Batch data
            
        Returns:
            List of generated summaries
        """
        predictions = []
        
        for i in range(len(batch["text"])):
            # Generate summary for each sample
            summary = self.model.generate_summary(
                text=batch["text"][i],
                image=batch["images"][i],
                max_length=self.config.evaluation.max_length,
                min_length=self.config.evaluation.min_length,
                num_beams=self.config.evaluation.num_beams,
                early_stopping=self.config.evaluation.early_stopping
            )
            predictions.append(summary)
        
        return predictions
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.save_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.training.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> None:
        """Main training loop."""
        logging.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate_epoch()
                
                # Update best validation loss
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, is_best=True)
                
                # Log metrics
                logging.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}, "
                           f"Val Loss = {val_metrics['val_loss']:.4f}")
                
                # Log to wandb
                if self.config.logging.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Early stopping
                if self.early_stopping(val_metrics['val_loss'], self.model):
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch)
        
        # Final evaluation
        if self.test_loader:
            logging.info("Running final evaluation on test set...")
            test_metrics = self.evaluate()
            logging.info(f"Test metrics: {test_metrics}")
            
            if self.config.logging.use_wandb:
                wandb.log(test_metrics)
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.config.logging.use_wandb:
            wandb.finish()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Returns:
            Dictionary containing test metrics
        """
        self.model.eval()
        predictions = []
        references = []
        texts = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                batch_predictions = self.generate_predictions(batch)
                predictions.extend(batch_predictions)
                references.extend(batch["summaries"])
                texts.extend(batch["text"])
        
        # Compute metrics
        metrics = evaluate_model_predictions(predictions, references, texts)
        
        return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-modal summarization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = MultiModalTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
