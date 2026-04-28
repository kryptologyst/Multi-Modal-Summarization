"""Tests for multi-modal summarization components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, EarlyStopping
from data import MultiModalDataset, create_data_loaders, collate_fn
from models import TextSummarizer, ImageCaptioner, MultiModalFusion, MultiModalSummarizer
from eval import MultiModalEvaluator, evaluate_model_predictions


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        set_seed(42)
        
        # Test that seeds are set
        torch.manual_seed(42)
        random_tensor = torch.randn(10)
        assert random_tensor is not None
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        # Test auto device selection
        device_auto = get_device("auto")
        assert device_auto is not None
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3)
        model = Mock()
        
        # Test improvement
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)
        
        # Test no improvement
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.5, model)
        assert early_stopping(0.5, model)  # Should trigger early stopping


class TestDataHandling:
    """Test data handling components."""
    
    def test_multimodal_dataset(self):
        """Test MultiModalDataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            sample_data = [
                {
                    "id": "test_1",
                    "text": "This is a test text.",
                    "image_path": "test_image.jpg",
                    "summary": "Test summary.",
                    "image_caption": "Test caption."
                }
            ]
            
            data_path = os.path.join(temp_dir, "test_data.json")
            with open(data_path, 'w') as f:
                json.dump(sample_data, f)
            
            # Create dataset
            dataset = MultiModalDataset(
                data_path=data_path,
                image_dir=temp_dir,
                max_text_length=100,
                image_size=224
            )
            
            assert len(dataset) == 1
            
            # Test getting item
            item = dataset[0]
            assert "text_input_ids" in item
            assert "image" in item
            assert "summary" in item
    
    def test_collate_fn(self):
        """Test collate function."""
        batch = [
            {
                "text_input_ids": torch.randn(10),
                "text_attention_mask": torch.ones(10),
                "image": torch.randn(3, 224, 224),
                "summary_input_ids": torch.randn(5),
                "summary_attention_mask": torch.ones(5),
                "id": "test_1",
                "text": "Test text",
                "image_path": "test.jpg",
                "summary": "Test summary",
                "image_caption": "Test caption"
            }
        ]
        
        collated = collate_fn(batch)
        
        assert "text_input_ids" in collated
        assert "images" in collated
        assert collated["text_input_ids"].shape[0] == 1


class TestModels:
    """Test model components."""
    
    def test_text_summarizer(self):
        """Test TextSummarizer initialization."""
        with patch('transformers.BartForConditionalGeneration.from_pretrained') as mock_model, \
             patch('transformers.BartTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_model.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            
            summarizer = TextSummarizer()
            assert summarizer is not None
    
    def test_image_captioner(self):
        """Test ImageCaptioner initialization."""
        with patch('transformers.BlipForConditionalGeneration.from_pretrained') as mock_model, \
             patch('transformers.BlipProcessor.from_pretrained') as mock_processor:
            
            mock_model.return_value = Mock()
            mock_processor.return_value = Mock()
            
            captioner = ImageCaptioner()
            assert captioner is not None
    
    def test_multimodal_fusion(self):
        """Test MultiModalFusion module."""
        fusion = MultiModalFusion(
            text_dim=1024,
            image_dim=768,
            hidden_dim=512,
            fusion_method="late"
        )
        
        text_features = torch.randn(2, 1024)
        image_features = torch.randn(2, 768)
        
        fused = fusion(text_features, image_features)
        assert fused.shape == (2, 512)
    
    def test_multimodal_fusion_cross_attention(self):
        """Test cross-attention fusion."""
        fusion = MultiModalFusion(
            text_dim=1024,
            image_dim=768,
            hidden_dim=512,
            fusion_method="cross_attention"
        )
        
        text_features = torch.randn(2, 1024)
        image_features = torch.randn(2, 768)
        
        fused = fusion(text_features, image_features)
        assert fused.shape == (2, 512)


class TestEvaluation:
    """Test evaluation components."""
    
    def test_multimodal_evaluator(self):
        """Test MultiModalEvaluator initialization."""
        evaluator = MultiModalEvaluator()
        assert evaluator is not None
    
    def test_rouge_scores(self):
        """Test ROUGE score computation."""
        evaluator = MultiModalEvaluator()
        
        predictions = ["This is a test summary."]
        references = ["This is a reference summary."]
        
        scores = evaluator.compute_rouge_scores(predictions, references)
        
        assert "rouge_rouge1" in scores
        assert "rouge_rouge2" in scores
        assert "rouge_rougeL" in scores
    
    def test_bleu_scores(self):
        """Test BLEU score computation."""
        evaluator = MultiModalEvaluator()
        
        predictions = ["This is a test summary."]
        references = ["This is a reference summary."]
        
        scores = evaluator.compute_bleu_scores(predictions, references)
        
        assert "bleu_1" in scores
        assert "bleu_2" in scores
        assert "bleu_3" in scores
        assert "bleu_4" in scores
    
    def test_evaluate_model_predictions(self):
        """Test comprehensive model evaluation."""
        predictions = ["This is a test summary."]
        references = ["This is a reference summary."]
        texts = ["This is a longer original text that needs to be summarized."]
        
        metrics = evaluate_model_predictions(predictions, references, texts)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestIntegration:
    """Integration tests."""
    
    def test_model_forward_pass(self):
        """Test complete model forward pass."""
        with patch('transformers.BartForConditionalGeneration.from_pretrained') as mock_bart, \
             patch('transformers.BartTokenizer.from_pretrained') as mock_bart_tokenizer, \
             patch('transformers.BlipForConditionalGeneration.from_pretrained') as mock_blip, \
             patch('transformers.BlipProcessor.from_pretrained') as mock_blip_processor:
            
            # Mock model components
            mock_bart.return_value = Mock()
            mock_bart_tokenizer.return_value = Mock()
            mock_blip.return_value = Mock()
            mock_blip_processor.return_value = Mock()
            
            # Create config
            config = {
                "model": {
                    "text_summarizer": {"name": "facebook/bart-large-cnn"},
                    "image_captioner": {"name": "Salesforce/blip-image-captioning-base"},
                    "fusion": {
                        "method": "late",
                        "hidden_dim": 512,
                        "attention_heads": 8
                    }
                }
            }
            
            # Create model
            model = MultiModalSummarizer(config)
            
            # Test forward pass
            batch_size = 2
            text_input_ids = torch.randint(0, 1000, (batch_size, 10))
            text_attention_mask = torch.ones(batch_size, 10)
            images = torch.randn(batch_size, 3, 224, 224)
            
            outputs = model(text_input_ids, text_attention_mask, images)
            
            assert "fused_features" in outputs
            assert "output_features" in outputs
            assert outputs["fused_features"].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__])
