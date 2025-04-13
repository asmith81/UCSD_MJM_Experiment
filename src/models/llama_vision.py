"""
Llama-3.2-11B-Vision model implementation for invoice data extraction.

This module implements the Llama Vision model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
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

class LlamaVisionModel:
    """Llama-3.2-11B-Vision model implementation."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Llama Vision model.
        
        Args:
            model_path: Path to model weights
            quantization: Bit width for quantization (4, 8, 16, 32)
            device: Device to run model on
        """
        self.model_path = Path(model_path)
        self.quantization = quantization
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model with specified quantization."""
        try:
            logger.info(f"Loading Llama Vision model with {self.quantization}-bit quantization")
            
            # Load tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = LlamaForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 16:
                self.model = LlamaForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 8:
                self.model = LlamaForCausalLM.from_pretrained(
                    str(self.model_path),
                    load_in_8bit=True,
                    device_map=self.device,
                    trust_remote_code=True
                )
            elif self.quantization == 4:
                self.model = LlamaForCausalLM.from_pretrained(
                    str(self.model_path),
                    load_in_4bit=True,
                    device_map=self.device,
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Llama Vision model: {str(e)}")
            raise
            
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
            
            # Prepare model inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Add image to inputs
            inputs["pixel_values"] = image
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7
                )
                
            # Decode output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse output
            parsed_output = parse_model_output(output_text, field_type)
            
            # Validate output
            if not validate_model_output(parsed_output, field_type):
                raise ValueError("Invalid model output")
                
            # Create response
            response: ModelResponse = {
                "output": parsed_output,
                "error": None,
                "processing_time": time.time() - start_time
            }
            
            # Return full result structure
            return {
                "test_parameters": {
                    "model": "llama_vision",
                    "quantization": self.quantization,
                    "prompt_strategy": prompt
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
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            response: ModelResponse = {
                "output": {
                    "raw_text": "",
                    "parsed_value": None,
                    "normalized_value": None
                },
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            return {
                "test_parameters": {
                    "model": "llama_vision",
                    "quantization": self.quantization,
                    "prompt_strategy": prompt
                },
                "model_response": response,
                "evaluation": {
                    "work_order_number": {
                        "normalized_match": False,
                        "cer": 0.0,
                        "error_category": str(e)
                    },
                    "total_cost": {
                        "normalized_match": False,
                        "cer": 0.0,
                        "error_category": str(e)
                    }
                }
            }

# Protocol-compatible wrapper functions
def load_model(model_name: str, quantization: int) -> LlamaVisionModel:
    """Load Llama Vision model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'llama_vision')
        quantization: Bit width for quantization (4, 8, 16, 32)
        
    Returns:
        Loaded LlamaVisionModel instance
        
    Raises:
        ValueError: If model_name is not 'llama_vision'
    """
    if model_name != "llama_vision":
        raise ValueError(f"Invalid model name for Llama Vision loader: {model_name}")
        
    return LlamaVisionModel(
        model_path="models/llama-3.2-11b-vision",
        quantization=quantization
    )

def process_image_wrapper(
    model: LlamaVisionModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Llama Vision model.
    
    Args:
        model: Loaded LlamaVisionModel instance
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
            "model": "llama_vision",
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