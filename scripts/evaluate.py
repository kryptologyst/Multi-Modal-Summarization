#!/usr/bin/env python3
"""Evaluation script for multi-modal summarization."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, load_config
from models import MultiModalSummarizer
from eval import MultiModalEvaluator, evaluate_model_predictions
from data import create_data_loaders


def evaluate_model(
    model: MultiModalSummarizer,
    test_loader,
    config: DictConfig,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model on test data.
    
    Args:
        model: Multi-modal summarization model
        test_loader: Test data loader
        config: Configuration object
        device: Device to run on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    predictions = []
    references = []
    texts = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate predictions
            batch_predictions = []
            for i in range(len(batch["text"])):
                summary = model.generate_summary(
                    text=batch["text"][i],
                    image=batch["images"][i],
                    max_length=config.evaluation.max_length,
                    min_length=config.evaluation.min_length,
                    num_beams=config.evaluation.num_beams,
                    early_stopping=config.evaluation.early_stopping
                )
                batch_predictions.append(summary)
            
            predictions.extend(batch_predictions)
            references.extend(batch["summaries"])
            texts.extend(batch["text"])
    
    # Compute comprehensive metrics
    metrics = evaluate_model_predictions(predictions, references, texts)
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal summarization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(config.device.type, config.device.fallback_to_cpu)
    
    # Initialize model
    model = MultiModalSummarizer(config)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(config)
    
    if test_loader is None:
        print("No test data available. Creating sample evaluation...")
        # Create sample evaluation
        sample_texts = [
            "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
            "Machine learning is revolutionizing healthcare through AI-powered diagnostic tools.",
            "Climate change poses significant challenges to global ecosystems."
        ]
        sample_summaries = [
            "The Eiffel Tower is a famous landmark in Paris.",
            "Machine learning transforms healthcare with AI diagnostics.",
            "Climate change threatens global ecosystems."
        ]
        
        metrics = evaluate_model_predictions(sample_summaries, sample_summaries, sample_texts)
    else:
        # Run evaluation
        metrics = evaluate_model(model, test_loader, config, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric:25}: {value:.4f}")
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
