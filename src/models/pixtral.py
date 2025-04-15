"""
Pixtral-12B model implementation for invoice data extraction.

This module implements the Pixtral-12B model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union, List
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import logging
from pathlib import Path
import time
from transformers import BitsAndBytesConfig
import json

from .common import (
    preprocess_image,
    parse_model_output,
    validate_model_output
)
from ..data_utils import DataConfig
from ..results_logging import ModelResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_cer(pred: str, true: str) -> float:
    """
    Calculate Character Error Rate between predicted and true values.
    
    Args:
        pred: Predicted string
        true: True string
        
    Returns:
        Character Error Rate (0.0 to 1.0)
    """
    if not true:
        return 1.0  # Maximum error if true value is empty
        
    # Levenshtein distance calculation
    if len(pred) < len(true):
        return calculate_cer(true, pred)
        
    if len(true) == 0:
        return len(pred)
        
    previous_row = range(len(true) + 1)
    for i, c1 in enumerate(pred):
        current_row = [i + 1]
        for j, c2 in enumerate(true):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1] / len(true)

def categorize_error(pred: str, true: str, field_type: str) -> str:
    """
    Categorize error type based on field and difference.
    
    Args:
        pred: Predicted value
        true: True value
        field_type: Type of field ('work_order' or 'total_cost')
        
    Returns:
        Error category string
    """
    if not pred or not true:
        return "missing_field"
        
    if field_type == "work_order":
        # Check for transposition
        if sorted(pred) == sorted(true) and pred != true:
            return "transposition"
            
        # Check for length differences
        if len(pred) < len(true):
            return "missing_character"
        if len(pred) > len(true):
            return "extra_character"
            
        return "wrong_character"
        
    elif field_type == "total_cost":
        # Check for currency symbol error
        if ('$' in pred) != ('$' in true):
            return "currency_error"
            
        # Check for decimal point errors
        if ('.' in pred) != ('.' in true):
            return "decimal_error"
            
        # Check for formatting errors
        if any(c in pred for c in ',.') != any(c in true for c in ',.'):
            return "formatting_error"
            
        return "digit_error"
        
    return "unknown_error"

class PixtralModel:
    """Pixtral-12B model implementation."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Pixtral model.
        
        Args:
            model_path: Path to model weights
            quantization: Bit width for quantization (4, 8, 16, 32)
            device: Device to run model on
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            ValueError: If quantization level is invalid
        """
        # Validate model path
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Validate quantization
        if quantization not in [4, 8, 16, 32]:
            raise ValueError(f"Invalid quantization level: {quantization}. Must be one of [4, 8, 16, 32]")
            
        self.quantization = quantization
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
        
        # Pre-process chat template format
        self.chat_template = self.processor.tokenizer.chat_template
        
    def _load_model(self) -> None:
        """Load model with specified quantization.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading Pixtral model with {self.quantization}-bit quantization")
            
            # Validate model directory structure
            required_files = ['config.json', 'model.safetensors.index.json']
            missing_files = [f for f in required_files if not (self.model_path / f).exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing required model files: {missing_files}")
            
            # Load processor with fast processing enabled
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True
            )
            
            # Set default dtype based on quantization
            if self.quantization in [4, 8, 16]:
                default_dtype = torch.bfloat16
            else:
                default_dtype = torch.float32
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map="cuda:0",  # Explicit GPU mapping
                    low_cpu_mem_usage=True,
                    torch_dtype=default_dtype
                )
            elif self.quantization == 16:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                    torch_dtype=default_dtype,
                    use_flash_attention_2=True
                )
            elif self.quantization == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="fp8"  # Using fp8 for better accuracy
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    use_flash_attention_2=True,
                    torch_dtype=torch.bfloat16  # Ensure model weights are in bfloat16
                )
            elif self.quantization == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    use_flash_attention_2=True,
                    torch_dtype=torch.bfloat16  # Ensure model weights are in bfloat16
                )
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            # Tie weights to avoid warning
            self.model.tie_weights()
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Pixtral model: {str(e)}")
            raise RuntimeError(f"Failed to load Pixtral model: {str(e)}")
            
    def process_image(
        self,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        prompt: str,
        field_type: str,
        config: DataConfig
    ) -> Dict[str, Any]:
        """Process image(s) and extract field value.
        
        Args:
            image_paths: Path(s) to input image(s)
            prompt: Prompt template for extraction
            field_type: Type of field to extract
            config: Data configuration
            
        Returns:
            Dictionary with model response and processing time
            
        Raises:
            ValueError: If processing fails
        """
        try:
            start_time = time.time()
            
            # Convert single path to list for consistent handling
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            # Load and preprocess images
            images = []
            for path in image_paths:
                # Convert path to Path object
                path = Path(path)
                
                # If path is relative, check if it already includes data/images/
                if not path.is_absolute():
                    path_str = str(path)
                    if path_str.startswith('data/images/'):
                        # If it already has data/images/, use it as is
                        path = Path(path_str)
                    else:
                        # Otherwise, join with image directory
                        path = config.image_dir / path
                
                # Load and preprocess image
                image = Image.open(path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images.append(image)
            
            # Format chat-style input with actual images
            chat = [{
                "role": "user",
                "content": [
                    {"type": "text", "content": prompt},
                    *[{"type": "image", "image": img} for img in images]
                ]
            }]
            
            # Apply chat template and process inputs in one step
            inputs = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Optimize input tensor conversion
            if self.quantization in [4, 8, 16]:
                # Convert all tensors to bfloat16 for consistency
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                # Keep attention mask as boolean
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'].bool()
            
            # Run inference with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Increased from 30 for better accuracy
                    num_return_sequences=1,
                    temperature=0.7,  # Increased from 0.3 for better diversity
                    do_sample=True,  # Enable sampling for better results
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                    early_stopping=True  # Stop when EOS is generated
                )
                
            # Decode output
            output_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Ensure output is not None or empty
            if not output_text:
                output_text = ""
                
            logger.debug(f"Raw model output: {output_text}")
            
            # Parse output
            parsed_value = parse_model_output(output_text, field_type)
            
            logger.debug(f"Parsed value: {parsed_value}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Structure result
            result = {
                'test_parameters': {
                    'model': 'pixtral',
                    'quantization': self.quantization
                },
                'model_response': {
                    'output': output_text,
                    'processing_time': processing_time
                }
            }
            
            logger.debug(f"Result structure: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

def download_pixtral_model(model_path: Path, repo_id: str) -> bool:
    """Download Pixtral model from HuggingFace.
    
    Args:
        model_path: Path where model should be downloaded
        repo_id: HuggingFace repository ID
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        RuntimeError: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
        
        if model_path.exists():
            logger.info(f"Model already exists at {model_path}")
            return True
            
        logger.info(f"Downloading Pixtral model from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Pixtral model: {str(e)}")
        return False

def load_model(model_name: str, quantization: int, models_dir: Path, config: dict) -> PixtralModel:
    """Load Pixtral model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'pixtral')
        quantization: Bit width for quantization (4, 8, 16, 32)
        models_dir: Path to models directory
        config: Model configuration from YAML
        
    Returns:
        Loaded PixtralModel instance
        
    Raises:
        ValueError: If model_name is not 'pixtral'
        FileNotFoundError: If model directory doesn't exist
    """
    if model_name != "pixtral":
        raise ValueError(f"Invalid model name for Pixtral loader: {model_name}")
        
    model_path = models_dir / "pixtral-12b"
    
    # Download model if needed
    if not model_path.exists():
        if not download_pixtral_model(model_path, config['repo_id']):
            raise RuntimeError(f"Failed to download Pixtral model to {model_path}")
        
    return PixtralModel(
        model_path=model_path,
        quantization=quantization
    )

def process_image_wrapper(
    model: PixtralModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Pixtral model.
    
    Args:
        model: Loaded PixtralModel instance
        prompt_template: Prompt template to use
        image_path: Path to input image
        field_type: Type of field to extract
        config: Data configuration
        
    Returns:
        Dictionary with test parameters and model response
    """
    # Get model response
    response = model.process_image(
        image_paths=image_path,
        prompt=prompt_template,
        field_type=field_type,
        config=config
    )
    
    # Update test parameters with prompt strategy
    response['test_parameters']['prompt_strategy'] = prompt_template
    
    # Ensure field type is set in model response
    response['model_response']['field_type'] = field_type
    
    return response 