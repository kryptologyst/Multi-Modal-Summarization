#!/usr/bin/env python3
"""Inference script for multi-modal summarization."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, load_config
from models import MultiModalSummarizer


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with multi-modal summarization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True,
                       help="Input text to summarize")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum length of generated summary")
    parser.add_argument("--min_length", type=int, default=20,
                       help="Minimum length of generated summary")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for generation")
    
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
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Preprocess image
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    image_tensor = preprocess_image(args.image)
    
    print("\n" + "="*50)
    print("MULTI-MODAL SUMMARIZATION")
    print("="*50)
    print(f"Input Text: {args.text}")
    print(f"Input Image: {args.image}")
    print("\nGenerating summary...")
    
    # Generate summary
    with torch.no_grad():
        summary = model.generate_summary(
            text=args.text,
            image=image_tensor,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=True
        )
    
    print(f"\nGenerated Summary: {summary}")
    
    # Also show individual components
    print("\n" + "-"*30)
    print("INDIVIDUAL COMPONENTS")
    print("-"*30)
    
    # Text summary
    text_summary = model.text_summarizer.generate_summary(
        args.text,
        max_length=args.max_length // 2,
        min_length=args.min_length // 2,
        temperature=args.temperature
    )
    print(f"Text Summary: {text_summary}")
    
    # Image caption
    image_caption = model.image_captioner.generate_caption(
        image_tensor,
        max_length=50,
        min_length=10,
        temperature=args.temperature
    )
    print(f"Image Caption: {image_caption}")


if __name__ == "__main__":
    main()
