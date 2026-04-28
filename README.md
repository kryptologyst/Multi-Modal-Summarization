# Multi-Modal Summarization

A research-ready implementation of multi-modal summarization that combines text summarization with image captioning to generate comprehensive summaries from both textual and visual content.

## Overview

This project implements a sophisticated multi-modal summarization system that:

- **Text Summarization**: Uses BART (Bidirectional and Auto-Regressive Transformers) for generating concise text summaries
- **Image Captioning**: Employs BLIP (Bootstrapping Language-Image Pre-training) for describing visual content
- **Multi-Modal Fusion**: Combines text and image features using various fusion strategies (early, late, cross-attention)
- **Comprehensive Evaluation**: Includes ROUGE, BLEU, BERTScore, METEOR, and CIDEr metrics
- **Interactive Demo**: Streamlit-based web interface for easy experimentation

## Features

### Core Capabilities
- **Multi-Modal Input Processing**: Handles text and image inputs simultaneously
- **Flexible Fusion Strategies**: Supports early, late, and cross-attention fusion methods
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Visualization Tools**: Attention heatmaps and feature distribution plots
- **Production Ready**: Clean code structure with proper error handling and logging

### Technical Highlights
- **Modern PyTorch 2.x**: Leverages latest PyTorch features and optimizations
- **Device Flexibility**: Automatic CUDA/MPS/CPU device selection with fallback
- **Reproducible**: Deterministic seeding and configuration management
- **Type Safety**: Full type hints and comprehensive documentation
- **Testing**: Unit tests and integration tests for reliability

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- MPS support (optional, for Apple Silicon)

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kryptologyst/Multi-Modal-Summarization.git
cd Multi-Modal-Summarization
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Quick Start

### 1. Basic Usage

```python
from src.models import MultiModalSummarizer
from src.utils import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize model
model = MultiModalSummarizer(config)

# Generate summary
text = "Your input text here..."
image = torch.randn(3, 224, 224)  # Your image tensor

summary = model.generate_summary(text, image)
print(f"Generated summary: {summary}")
```

### 2. Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

Then open your browser to `http://localhost:8501` and:
1. Upload an image
2. Enter text to summarize
3. Adjust generation parameters
4. Generate multi-modal summary

### 3. Training

Train your own model:

```bash
python scripts/train.py --config configs/default.yaml
```

## Project Structure

```
multimodal-summarization/
├── src/                          # Source code
│   ├── data/                     # Data handling and preprocessing
│   ├── models/                   # Model implementations
│   ├── eval/                     # Evaluation metrics
│   ├── viz/                      # Visualization utilities
│   └── utils/                   # Utility functions
├── configs/                      # Configuration files
├── scripts/                      # Training and evaluation scripts
├── demo/                         # Interactive demos
├── tests/                        # Unit and integration tests
├── data/                         # Data directory
│   ├── images/                   # Image files
│   ├── text/                     # Text files
│   └── annotations.json          # Data annotations
├── assets/                       # Generated visualizations
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration sections:

### Model Configuration
```yaml
model:
  text_summarizer:
    name: "facebook/bart-large-cnn"
    max_length: 100
    min_length: 30
  
  image_captioner:
    name: "Salesforce/blip-image-captioning-base"
    max_length: 50
    min_length: 10
  
  fusion:
    method: "late"  # Options: early, late, cross_attention
    hidden_dim: 512
    attention_heads: 8
```

### Training Configuration
```yaml
training:
  learning_rate: 1e-4
  weight_decay: 0.01
  num_epochs: 10
  batch_size: 8
  use_amp: true  # Mixed precision training
```

## Data Format

The project expects data in JSON format:

```json
[
  {
    "id": "sample_1",
    "text": "Your input text here...",
    "image_path": "path/to/image.jpg",
    "summary": "Reference summary...",
    "image_caption": "Reference image caption..."
  }
]
```

### Sample Data Generation

If no data is provided, the system automatically generates sample data for demonstration purposes.

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

### Automatic Metrics
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU**: BLEU-1 through BLEU-4
- **BERTScore**: Precision, Recall, F1
- **METEOR**: Semantic similarity
- **CIDEr**: Consensus-based evaluation

### Summarization-Specific Metrics
- **Compression Ratio**: Summary length vs. original text length
- **Extractive Coverage**: How much of the summary comes from the original text
- **Abstractiveness Score**: Measure of abstractive vs. extractive summarization

## Model Architecture

### Text Summarization
- **Backbone**: BART (Bidirectional and Auto-Regressive Transformers)
- **Architecture**: Encoder-decoder with cross-attention
- **Pre-training**: Large-scale denoising autoencoder

### Image Captioning
- **Backbone**: BLIP (Bootstrapping Language-Image Pre-training)
- **Architecture**: Vision-language transformer
- **Features**: Unified vision-language understanding

### Multi-Modal Fusion
- **Early Fusion**: Concatenate features before processing
- **Late Fusion**: Process modalities separately, then combine
- **Cross-Attention**: Attention-based interaction between modalities

## Training

### Basic Training
```bash
python scripts/train.py --config configs/default.yaml
```

### Resume Training
```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_5.pt
```

### Training Features
- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Stops training when validation loss stops improving
- **Checkpointing**: Saves model checkpoints at regular intervals
- **Logging**: Comprehensive logging with TensorBoard and Weights & Biases support

## Evaluation

### Run Evaluation
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

### Evaluation Output
The evaluation produces:
- **Quantitative Metrics**: All computed evaluation scores
- **Qualitative Results**: Sample predictions with visualizations
- **Leaderboard**: Comparison across different configurations

## Visualization

The project includes comprehensive visualization tools:

### Attention Visualization
- **Text Attention**: Heatmaps showing which words the model focuses on
- **Image Attention**: Visual attention overlays on images
- **Cross-Modal Attention**: How text and image features interact

### Training Curves
- **Loss Curves**: Training and validation loss over time
- **Metric Curves**: Evaluation metrics during training
- **Learning Rate**: Learning rate schedule visualization

### Feature Analysis
- **Feature Distributions**: Distribution of different feature types
- **Dimensionality Reduction**: t-SNE/UMAP plots of learned features

## API Reference

### Core Classes

#### `MultiModalSummarizer`
Main model class combining text summarization and image captioning.

```python
model = MultiModalSummarizer(config)
summary = model.generate_summary(text, image)
```

#### `MultiModalEvaluator`
Comprehensive evaluation with multiple metrics.

```python
evaluator = MultiModalEvaluator()
metrics = evaluator.compute_all_metrics(predictions, references)
```

#### `MultiModalVisualizer`
Visualization utilities for results and attention.

```python
visualizer = MultiModalVisualizer()
fig = visualizer.plot_attention_heatmap(attention_weights, tokens)
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with proper tests
4. **Run tests**: `pytest tests/`
5. **Format code**: `black src/ tests/` and `ruff check src/ tests/`
6. **Submit a pull request**

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/ tests/
```

## Safety and Limitations

### Important Disclaimers

**Research/Educational Use Only**: This project is designed for research and educational purposes. Generated summaries may not be accurate or complete.

**Not for Critical Applications**: Do not use for critical decision-making, medical diagnosis, legal advice, or other high-stakes applications.

**Bias and Fairness**: The models may exhibit biases present in training data. Always evaluate outputs critically.

### Safety Features

- **Content Filtering**: Basic safety filters for generated content
- **Input Validation**: Validation of input text and images
- **Error Handling**: Comprehensive error handling and logging
- **Resource Limits**: Configurable limits on input size and generation length

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multimodal_summarization,
  title={Multi-Modal Summarization: Combining Text Summarization with Image Captioning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-Modal-Summarization}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face Transformers**: For the excellent transformer implementations
- **BART**: Facebook AI Research for the BART model
- **BLIP**: Salesforce Research for the BLIP model
- **PyTorch Team**: For the amazing deep learning framework# Multi-Modal-Summarization
