#!/usr/bin/env python3
"""
Project 937: Multi-modal Summarization - Simple Example

This is a simplified example demonstrating the basic functionality of the 
multi-modal summarization system. For the full implementation with advanced
features, training capabilities, and comprehensive evaluation, see the 
modernized codebase in the src/ directory.

This example shows:
1. Text summarization using BART
2. Image captioning using BLIP  
3. Simple combination of both modalities
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


def summarize_text(text: str) -> str:
    """Summarize text using BART model.
    
    Args:
        text: Input text to summarize
        
    Returns:
        Generated summary
    """
    text_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = text_summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def caption_image(image_path: str) -> str:
    """Generate caption for image using BLIP model.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Generated caption
    """
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_path)
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    """Main function demonstrating multi-modal summarization."""
    print("=" * 60)
    print("Multi-Modal Summarization - Simple Example")
    print("=" * 60)
    
    # Example text input
    text_input = """
    The Eiffel Tower is one of the most famous landmarks in Paris, France. 
    It was designed by engineer Gustave Eiffel and completed in 1889. 
    The tower stands at 324 meters and was initially met with skepticism, 
    but it became an iconic symbol of France's ingenuity and elegance. 
    Today, it attracts millions of tourists every year who come to see 
    its breathtaking views of the city.
    """
    
    print("Step 1: Text Summarization")
    print("-" * 30)
    print(f"Original Text: {text_input.strip()}")
    
    try:
        text_summary = summarize_text(text_input)
        print(f"Text Summary: {text_summary}")
    except Exception as e:
        print(f"Error in text summarization: {e}")
        text_summary = "Text summarization failed"
    
    print("\nStep 2: Image Captioning")
    print("-" * 30)
    
    # Try to use a sample image, fallback to placeholder
    image_path = "data/images/sample_images/eiffel_tower.jpg"
    
    try:
        if Path(image_path).exists():
            image_caption = caption_image(image_path)
            print(f"Image Caption: {image_caption}")
        else:
            print("Sample image not found. Using placeholder.")
            image_caption = "A tall iron lattice tower structure in Paris, France, known as the Eiffel Tower."
    except Exception as e:
        print(f"Error in image captioning: {e}")
        image_caption = "Image captioning failed"
    
    print("\nStep 3: Multi-Modal Summary")
    print("-" * 30)
    combined_summary = f"Text Summary: {text_summary}\nImage Caption: {image_caption}"
    print(f"Combined Multi-modal Summary:\n{combined_summary}")
    
    print("\n" + "=" * 60)
    print("For advanced features, training, and evaluation, see:")
    print("- src/ directory: Full implementation")
    print("- scripts/train.py: Training script")
    print("- demo/streamlit_app.py: Interactive demo")
    print("- README.md: Complete documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()

