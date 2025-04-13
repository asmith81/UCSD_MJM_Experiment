"""
Doctr model implementation for invoice data extraction.

This module implements the Doctr model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union
import torch
from doctr.models import ocr_predictor
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

class DoctrModel:
    """Doctr model implementation."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Doctr model.
        
        Args:
            model_path: Path to model weights
            quantization: Bit width for quantization (4, 8, 16, 32)
            device: Device to run model on
        """
        self.model_path = Path(model_path)
        self.quantization = quantization
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model with specified quantization."""
        try:
            logger.info(f"Loading Doctr model with {self.quantization}-bit quantization")
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True,
                    device=self.device
                )
            elif self.quantization == 16:
                self.model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True,
                    device=self.device
                )
                self.model = self.model.half()  # Convert to float16
            elif self.quantization == 8:
                self.model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True,
                    device=self.device
                )
                # Apply 8-bit quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            elif self.quantization == 4:
                raise ValueError("4-bit quantization not supported for Doctr model")
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Doctr model: {str(e)}")
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
            prompt: Prompt template for extraction (not used for Doctr)
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
            
            # Run inference
            with torch.no_grad():
                result = self.model([image])
                
            # Extract text from result
            output_text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        output_text += line.value + " "
            
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
                    "model": "doctr",
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
                    "model": "doctr",
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
def load_model(model_name: str, quantization: int) -> DoctrModel:
    """Load Doctr model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'doctr')
        quantization: Bit width for quantization (4, 8, 16, 32)
        
    Returns:
        Loaded DoctrModel instance
        
    Raises:
        ValueError: If model_name is not 'doctr'
    """
    if model_name != "doctr":
        raise ValueError(f"Invalid model name for Doctr loader: {model_name}")
        
    return DoctrModel(
        model_path="models/doctr",
        quantization=quantization
    )

def process_image_wrapper(
    model: DoctrModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Doctr model.
    
    Args:
        model: Loaded DoctrModel instance
        prompt_template: Prompt template to use (not used for Doctr)
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
            "model": "doctr",
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