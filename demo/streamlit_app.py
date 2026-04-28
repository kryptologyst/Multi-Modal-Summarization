"""Streamlit demo for multi-modal summarization."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import logging
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, load_config
from models import MultiModalSummarizer
from viz import MultiModalVisualizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model(config_path: str = "configs/default.yaml"):
    """Load the multi-modal summarization model.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded model and configuration
    """
    try:
        config = load_config(config_path)
        device = get_device(config.device.type, config.device.fallback_to_cpu)
        
        model = MultiModalSummarizer(config)
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, config, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input.
    
    Args:
        image: PIL Image
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def generate_summary_with_visualization(
    model: MultiModalSummarizer,
    text: str,
    image: torch.Tensor,
    config: dict,
    device: torch.device
) -> tuple:
    """Generate summary with attention visualization.
    
    Args:
        model: Multi-modal summarization model
        text: Input text
        image: Input image tensor
        config: Model configuration
        device: Device to run on
        
    Returns:
        Tuple of (summary, attention_data)
    """
    with torch.no_grad():
        # Generate summary
        summary = model.generate_summary(
            text=text,
            image=image.to(device),
            max_length=config.evaluation.max_length,
            min_length=config.evaluation.min_length,
            num_beams=config.evaluation.num_beams,
            early_stopping=config.evaluation.early_stopping
        )
        
        # Get attention weights for visualization
        # This is a simplified version - in practice you'd want more sophisticated attention extraction
        attention_data = None
        
        return summary, attention_data


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Modal Summarization",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📝 Multi-Modal Summarization")
    st.markdown("""
    This demo showcases multi-modal summarization that combines text summarization with image captioning.
    Upload an image and provide some text, and the model will generate a comprehensive summary.
    """)
    
    # Safety disclaimer
    st.warning("""
    **Disclaimer**: This is a research/educational tool. Generated summaries may not be accurate or complete.
    Do not use for critical decision-making or professional purposes.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model loading
    with st.spinner("Loading model..."):
        model, config, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the configuration and try again.")
        st.stop()
    
    # Display model info
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.info(f"Device: {device}")
    st.sidebar.info(f"Fusion method: {config.model.fusion.method}")
    
    # Input parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider("Max Summary Length", 50, 200, 150)
    min_length = st.sidebar.slider("Min Summary Length", 10, 50, 20)
    num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        text_input = st.text_area(
            "Enter text to summarize:",
            value="The Eiffel Tower is one of the most famous landmarks in Paris, France. It was designed by engineer Gustave Eiffel and completed in 1889. The tower stands at 324 meters and was initially met with skepticism, but it became an iconic symbol of France's ingenuity and elegance. Today, it attracts millions of tourists every year who come to see its breathtaking views of the city.",
            height=200,
            help="Enter the text you want to summarize"
        )
        
        st.subheader("🖼️ Input Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to include in the multi-modal summary"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            image_tensor = preprocess_image(image)
        else:
            # Use default placeholder image
            st.info("No image uploaded. Using placeholder image.")
            # Create a simple colored image as placeholder
            image = Image.new('RGB', (224, 224), color=(100, 150, 200))
            st.image(image, caption="Placeholder Image", use_column_width=True)
            image_tensor = preprocess_image(image)
    
    with col2:
        st.subheader("📊 Generated Summary")
        
        if st.button("Generate Summary", type="primary"):
            if not text_input.strip():
                st.error("Please enter some text to summarize.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        # Generate summary
                        summary, attention_data = generate_summary_with_visualization(
                            model=model,
                            text=text_input,
                            image=image_tensor,
                            config=config,
                            device=device
                        )
                        
                        # Display summary
                        st.success("Summary generated successfully!")
                        st.markdown(f"**Generated Summary:**")
                        st.write(summary)
                        
                        # Display individual components
                        st.markdown("---")
                        st.subheader("🔍 Individual Components")
                        
                        # Text summary
                        text_summary = model.text_summarizer.generate_summary(
                            text_input,
                            max_length=max_length // 2,
                            min_length=min_length // 2,
                            temperature=temperature
                        )
                        st.markdown("**Text Summary:**")
                        st.write(text_summary)
                        
                        # Image caption
                        image_caption = model.image_captioner.generate_caption(
                            image_tensor,
                            max_length=50,
                            min_length=10,
                            temperature=temperature
                        )
                        st.markdown("**Image Caption:**")
                        st.write(image_caption)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        logger.error(f"Error in summary generation: {e}")
        
        # Example outputs
        st.subheader("📋 Example Outputs")
        with st.expander("Show Example Summaries"):
            st.markdown("""
            **Example 1:**
            - **Input Text:** "Machine learning is revolutionizing healthcare..."
            - **Input Image:** Medical scan image
            - **Generated Summary:** "Machine learning transforms healthcare through AI-powered diagnostic tools and medical imaging analysis, enabling faster and more accurate patient care."
            
            **Example 2:**
            - **Input Text:** "Climate change poses significant challenges..."
            - **Input Image:** Environmental data visualization
            - **Generated Summary:** "Climate change presents urgent environmental challenges requiring immediate action, as visualized through rising temperature trends and ecosystem impacts."
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Multi-Modal Summarization Demo | Built with Streamlit and PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
