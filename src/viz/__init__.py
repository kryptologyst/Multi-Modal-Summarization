"""Visualization utilities for multi-modal summarization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import seaborn as sns
import logging


class MultiModalVisualizer:
    """Visualization utilities for multi-modal summarization results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
        
        logging.info("Initialized multi-modal visualizer")
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot attention heatmap for text tokens.
        
        Args:
            attention_weights: Attention weights tensor [num_heads, seq_len, seq_len]
            tokens: List of token strings
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()
        
        # Create heatmap
        im = ax.imshow(avg_attention, cmap='Blues', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_image_with_attention(
        self,
        image: torch.Tensor,
        attention_weights: torch.Tensor,
        title: str = "Image with Attention",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot image with attention overlay.
        
        Args:
            image: Image tensor [C, H, W]
            attention_weights: Attention weights [H, W]
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert image tensor to numpy
        if image.dim() == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Normalize image
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Plot original image
        ax1.imshow(img_np)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Plot attention map
        attention_np = attention_weights.cpu().numpy()
        im2 = ax2.imshow(attention_np, cmap='hot', interpolation='nearest')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Plot overlay
        ax3.imshow(img_np)
        ax3.imshow(attention_np, cmap='hot', alpha=0.5, interpolation='nearest')
        ax3.set_title("Image + Attention")
        ax3.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of metrics across different models/configurations.
        
        Args:
            metrics_data: Dictionary with model names as keys and metrics as values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not metrics_data:
            logging.warning("No metrics data provided")
            return plt.figure()
        
        # Extract metrics and models
        models = list(metrics_data.keys())
        metrics = list(next(iter(metrics_data.values())).keys())
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            values = [metrics_data[model].get(metric, 0.0) for model in models]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        title: str = "Training Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_plots = 1 + (1 if val_losses else 0) + len(train_metrics or {})
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot training loss
        ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
        
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        
        ax.set_title('Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot metrics
        if train_metrics:
            for metric_name, metric_values in train_metrics.items():
                if plot_idx >= n_plots:
                    break
                    
                ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
                ax.plot(epochs, metric_values, 'g-', label=f'Training {metric_name}', alpha=0.7)
                
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 'orange', 
                           label=f'Validation {metric_name}', alpha=0.7)
                
                ax.set_title(f'{metric_name.replace("_", " ").title()} Curves')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_grid(
        self,
        images: List[torch.Tensor],
        texts: List[str],
        summaries: List[str],
        captions: List[str],
        title: str = "Multi-modal Summarization Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create a grid showing images, texts, summaries, and captions.
        
        Args:
            images: List of image tensors
            texts: List of input texts
            summaries: List of generated summaries
            captions: List of image captions
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_samples = len(images)
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Plot image
            ax_img = axes[i, 0]
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            ax_img.imshow(img_np)
            ax_img.set_title(f"Sample {i+1}")
            ax_img.axis('off')
            
            # Plot text information
            ax_text = axes[i, 1]
            ax_text.axis('off')
            
            # Add text content
            text_content = f"""
            Original Text:
            {texts[i][:200]}{'...' if len(texts[i]) > 200 else ''}
            
            Generated Summary:
            {summaries[i]}
            
            Image Caption:
            {captions[i]}
            """
            
            ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        fused_features: torch.Tensor,
        title: str = "Feature Distributions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distributions of different feature types.
        
        Args:
            text_features: Text features tensor
            image_features: Image features tensor
            fused_features: Fused features tensor
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert to numpy
        text_np = text_features.cpu().numpy().flatten()
        image_np = image_features.cpu().numpy().flatten()
        fused_np = fused_features.cpu().numpy().flatten()
        
        # Plot distributions
        axes[0].hist(text_np, bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Text Features Distribution')
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(image_np, bins=50, alpha=0.7, color='green')
        axes[1].set_title('Image Features Distribution')
        axes[1].set_xlabel('Feature Value')
        axes[1].set_ylabel('Frequency')
        
        axes[2].hist(fused_np, bins=50, alpha=0.7, color='red')
        axes[2].set_title('Fused Features Distribution')
        axes[2].set_xlabel('Feature Value')
        axes[2].set_ylabel('Frequency')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def save_visualization(fig: plt.Figure, save_path: str) -> None:
    """Save visualization to file.
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save the figure
    """
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved visualization to {save_path}")


def create_attention_visualization(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_dir: str = "assets"
) -> str:
    """Create and save attention visualization.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens
        save_dir: Directory to save visualization
        
        Returns:
            Path to saved visualization
    """
    visualizer = MultiModalVisualizer()
    fig = visualizer.plot_attention_heatmap(attention_weights, tokens)
    
    save_path = f"{save_dir}/attention_heatmap.png"
    save_visualization(fig, save_path)
    
    return save_path
