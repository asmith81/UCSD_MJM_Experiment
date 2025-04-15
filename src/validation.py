"""
Result validation module.

This module provides functionality for validating test results across different models.
"""

from typing import Dict, Any, List
import logging
from pathlib import Path
from .results_logging import evaluate_model_output
from .data_utils import GroundTruthData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_test_matrix(
    test_matrix: dict,
    supported_quant_levels: List[int],
    available_prompt_types: List[str],
    data_dir: Path
) -> None:
    """Validate test matrix structure and content.
    
    Args:
        test_matrix: Test matrix dictionary
        supported_quant_levels: List of supported quantization levels
        available_prompt_types: List of available prompt types
        data_dir: Path to data directory
        
    Raises:
        ValueError: If test matrix is invalid
    """
    if not isinstance(test_matrix, dict):
        raise ValueError("Test matrix must be a dictionary")
        
    if 'test_cases' not in test_matrix:
        raise ValueError("Test matrix must contain 'test_cases' key")
        
    for case in test_matrix['test_cases']:
        # Validate required fields
        required_fields = ['model_name', 'quant_level', 'prompt_type', 'image_number', 'field_type', 'image_path']
        missing_fields = [field for field in required_fields if field not in case]
        if missing_fields:
            raise ValueError(f"Test case missing required fields: {missing_fields}")
            
        # Validate quantization level
        if case['quant_level'] not in supported_quant_levels:
            raise ValueError(f"Unsupported quantization level: {case['quant_level']}")
            
        # Validate prompt type
        if case['prompt_type'] not in available_prompt_types:
            raise ValueError(f"Unsupported prompt type: {case['prompt_type']}")
            
        # Validate image path exists
        image_path = Path(case['image_path'])
        if not image_path.is_absolute():
            image_path = data_dir / image_path
            
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")

def validate_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate test results.
    
    Args:
        result: Dictionary containing test results
        
    Returns:
        Validated and normalized results
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check required fields
        required_fields = ['test_parameters', 'model_response']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Result missing required fields: {missing_fields}")
            
        # Validate test parameters
        if 'model' not in result['test_parameters']:
            raise ValueError("Missing model name in test parameters")
        if 'quantization' not in result['test_parameters']:
            raise ValueError("Missing quantization level in test parameters")
            
        # Validate model response - allow both direct and chat-style formats
        if 'output' not in result['model_response'] or not result['model_response']['output']:
            # Try to find output in chat-style format
            if 'parsed_value' in result['model_response'] and result['model_response']['parsed_value']:
                result['model_response']['output'] = result['model_response']['parsed_value']
            else:
                # If no valid output found, set a default empty string
                result['model_response']['output'] = ""
                
        # Add default processing time if missing
        if 'processing_time' not in result['model_response']:
            result['model_response']['processing_time'] = 0.0
            logger.warning("Processing time not provided, using default value of 0.0")
            
        # Get ground truth from test parameters
        ground_truth = result['test_parameters'].get('ground_truth', {})
        field_type = result['test_parameters'].get('field_type', 'both')
        
        # Evaluate results if not already evaluated
        if 'evaluation' not in result:
            result['evaluation'] = evaluate_model_output(
                result['model_response']['output'],
                ground_truth,
                field_type
            )
            
        # Ensure evaluation fields have required subfields
        for field in ['work_order_number', 'total_cost']:
            if 'normalized_match' not in result['evaluation'][field]:
                raise ValueError(f"Missing normalized_match in {field} evaluation")
            if 'cer' not in result['evaluation'][field]:
                raise ValueError(f"Missing CER in {field} evaluation")
                
        return result
        
    except Exception as e:
        logger.error(f"Error validating results: {str(e)}")
        raise 