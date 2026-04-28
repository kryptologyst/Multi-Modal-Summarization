"""Data handling and preprocessing for multi-modal summarization."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import logging


class MultiModalDataset(Dataset):
    """Dataset class for multi-modal summarization data."""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer_name: str = "facebook/bart-large-cnn",
        max_text_length: int = 1024,
        image_size: int = 224,
        image_mean: List[float] = [0.485, 0.456, 0.406],
        image_std: List[float] = [0.229, 0.224, 0.225],
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to JSON file containing data
            image_dir: Directory containing images
            tokenizer_name: Name of the tokenizer to use
            max_text_length: Maximum length for text sequences
            image_size: Size to resize images to
            image_mean: Mean values for image normalization
            image_std: Standard deviation values for image normalization
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.max_text_length = max_text_length
        
        # Load data
        self.data = self._load_data()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
        
        logging.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file.
        
        Returns:
            List of data samples
        """
        if not os.path.exists(self.data_path):
            logging.warning(f"Data file {self.data_path} not found. Creating sample data.")
            return self._create_sample_data()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("Invalid data format. Expected list or dict with 'data' key.")
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample data for demonstration purposes.
        
        Returns:
            List of sample data
        """
        sample_data = [
            {
                "id": "sample_1",
                "text": "The Eiffel Tower is one of the most famous landmarks in Paris, France. It was designed by engineer Gustave Eiffel and completed in 1889. The tower stands at 324 meters and was initially met with skepticism, but it became an iconic symbol of France's ingenuity and elegance. Today, it attracts millions of tourists every year who come to see its breathtaking views of the city.",
                "image_path": "sample_images/eiffel_tower.jpg",
                "summary": "The Eiffel Tower is a famous 324-meter landmark in Paris, designed by Gustave Eiffel in 1889, symbolizing French ingenuity and attracting millions of tourists annually.",
                "image_caption": "A tall iron lattice tower structure in Paris, France, known as the Eiffel Tower."
            },
            {
                "id": "sample_2", 
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. It has applications in various fields including computer vision, natural language processing, robotics, and data analysis.",
                "image_path": "sample_images/ml_concept.jpg",
                "summary": "Machine learning is an AI subset using algorithms to improve performance through experience, with applications in computer vision, NLP, robotics, and data analysis.",
                "image_caption": "A conceptual diagram showing machine learning algorithms and data flow."
            },
            {
                "id": "sample_3",
                "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities have been the main driver of climate change since the 1800s, primarily due to burning fossil fuels which generates greenhouse gas emissions.",
                "image_path": "sample_images/climate_change.jpg", 
                "summary": "Climate change involves long-term global temperature shifts, primarily driven by human activities like fossil fuel burning since the 1800s.",
                "image_caption": "A visual representation of Earth with climate change indicators and temperature variations."
            }
        ]
        
        # Create sample images directory
        os.makedirs(os.path.join(self.image_dir, "sample_images"), exist_ok=True)
        
        # Create placeholder images (colored rectangles for demo)
        for i, sample in enumerate(sample_data):
            img_path = os.path.join(self.image_dir, sample["image_path"])
            if not os.path.exists(img_path):
                # Create a simple colored image as placeholder
                img = Image.new('RGB', (224, 224), color=(100 + i*50, 150 + i*30, 200 - i*40))
                img.save(img_path)
        
        return sample_data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        sample = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, sample["image_path"])
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transforms(image)
        except Exception as e:
            logging.warning(f"Error loading image {image_path}: {e}")
            # Create a placeholder image
            image_tensor = torch.zeros(3, 224, 224)
        
        # Tokenize text
        text_encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize summary
        summary_encoding = self.tokenizer(
            sample["summary"],
            max_length=150,  # Shorter for summaries
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "id": sample["id"],
            "text": sample["text"],
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
            "image": image_tensor,
            "image_path": sample["image_path"],
            "summary": sample["summary"],
            "summary_input_ids": summary_encoding["input_ids"].squeeze(0),
            "summary_attention_mask": summary_encoding["attention_mask"].squeeze(0),
            "image_caption": sample.get("image_caption", ""),
        }


def create_data_loaders(
    config: Dict[str, Any],
    tokenizer_name: str = "facebook/bart-large-cnn"
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config["data"]
    
    # Create datasets
    train_dataset = MultiModalDataset(
        data_path=data_config["train_data"],
        image_dir=data_config.get("image_dir", "data/images"),
        tokenizer_name=tokenizer_name,
        max_text_length=data_config["max_text_length"],
        image_size=data_config["image_size"],
        image_mean=data_config["image_mean"],
        image_std=data_config["image_std"],
    )
    
    val_dataset = None
    test_dataset = None
    
    if data_config.get("val_data") and os.path.exists(data_config["val_data"]):
        val_dataset = MultiModalDataset(
            data_path=data_config["val_data"],
            image_dir=data_config.get("image_dir", "data/images"),
            tokenizer_name=tokenizer_name,
            max_text_length=data_config["max_text_length"],
            image_size=data_config["image_size"],
            image_mean=data_config["image_mean"],
            image_std=data_config["image_std"],
        )
    
    if data_config.get("test_data") and os.path.exists(data_config["test_data"]):
        test_dataset = MultiModalDataset(
            data_path=data_config["test_data"],
            image_dir=data_config.get("image_dir", "data/images"),
            tokenizer_name=tokenizer_name,
            max_text_length=data_config["max_text_length"],
            image_size=data_config["image_size"],
            image_mean=data_config["image_mean"],
            image_std=data_config["image_std"],
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            num_workers=data_config["num_workers"],
            pin_memory=True,
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            num_workers=data_config["num_workers"],
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data dictionary
    """
    # Stack tensors
    text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
    text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    summary_input_ids = torch.stack([item["summary_input_ids"] for item in batch])
    summary_attention_mask = torch.stack([item["summary_attention_mask"] for item in batch])
    
    return {
        "ids": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "images": images,
        "image_paths": [item["image_path"] for item in batch],
        "summaries": [item["summary"] for item in batch],
        "summary_input_ids": summary_input_ids,
        "summary_attention_mask": summary_attention_mask,
        "image_captions": [item["image_caption"] for item in batch],
    }
