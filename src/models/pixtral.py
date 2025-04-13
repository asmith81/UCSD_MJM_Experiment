"""
Pixtral-12B model implementation for invoice data extraction.

This module implements the Pixtral-12B model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from pathlib import Path
import time

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
        
    def _load_model(self) -> None:
        """Load model with specified quantization.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading Pixtral model with {self.quantization}-bit quantization")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 16:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 8:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    load_in_8bit=True,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 4:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    load_in_4bit=True,
                    device_map=self.device,
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Pixtral model: {str(e)}")
            raise RuntimeError(f"Failed to load Pixtral model: {str(e)}")
            
    def process_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        field_type: str,
        config: DataConfig
    ) -> Dict[str, Any]:
        """Process image and extract field value.
        
        Args:
            image_path: Path to input image
            prompt: Prompt template for extraction
            field_type: Type of field to extract
            config: Data configuration
            
        Returns:
            Dictionary with test parameters and model response
            
        Raises:
            ValueError: If processing fails
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            image = preprocess_image(Path(image_path), config)
            
            # Prepare inputs using processor
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7
                )
                
            # Decode output
            output_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse output
            parsed_value = parse_model_output(output_text, field_type)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Structure result according to validation requirements
            result = {
                'test_parameters': {
                    'model': 'pixtral',
                    'quantization': self.quantization
                },
                'model_response': {
                    'output': output_text,
                    'parsed_value': parsed_value,
                    'processing_time': processing_time
                },
                'evaluation': {
                    'work_order_number': {
                        'raw_string_match': False,
                        'normalized_match': False,
                        'cer': 0.0,
                        'error_category': 'not_evaluated'
                    },
                    'total_cost': {
                        'raw_string_match': False,
                        'normalized_match': False,
                        'cer': 0.0,
                        'error_category': 'not_evaluated'
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

# Protocol-compatible wrapper functions
def load_model(model_name: str, quantization: int) -> PixtralModel:
    """Load Pixtral model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'pixtral')
        quantization: Bit width for quantization (4, 8, 16, 32)
        
    Returns:
        Loaded PixtralModel instance
        
    Raises:
        ValueError: If model_name is not 'pixtral'
    """
    if model_name != "pixtral":
        raise ValueError(f"Invalid model name for Pixtral loader: {model_name}")
        
    return PixtralModel(
        model_path="models/pixtral-12b",
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
    response = model.process_image(
        image_path=image_path,
        prompt=prompt_template,
        field_type=field_type,
        config=config
    )
    
    return {
        "test_parameters": {
            "model": "pixtral",
            "quantization": model.quantization,
            "prompt_strategy": prompt_template
        },
        "model_response": response,
        "evaluation": {
            "work_order_number": {
                "normalized_match": False,
                "cer": 0.0,
                "error_category": None
            },
            "total_cost": {
                "normalized_match": False,
                "cer": 0.0,
                "error_category": None
            }
        }
    } 