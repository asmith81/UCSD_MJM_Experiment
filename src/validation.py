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

def validate_test_matrix(test_matrix: Dict[str, Any], supported_quant_levels: List[int], 
                        available_prompt_types: List[str], data_dir: Path) -> bool:
    """
    Validate test matrix structure and content.
    
    Args:
        test_matrix: Test matrix dictionary
        supported_quant_levels: List of supported quantization levels
        available_prompt_types: List of available prompt types
        data_dir: Path to data directory
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check required fields
        if 'test_cases' not in test_matrix:
            raise ValueError("Test matrix must contain 'test_cases' array")
            
        # Validate each test case
        for i, test_case in enumerate(test_matrix['test_cases']):
            # Check required fields
            required_fields = ['model_name', 'prompt_type', 'quant_level', 'image_path']
            missing_fields = [field for field in required_fields if field not in test_case]
            if missing_fields:
                raise ValueError(f"Test case {i} missing required fields: {missing_fields}")
                
            # Validate quantization level
            if test_case['quant_level'] not in supported_quant_levels:
                raise ValueError(f"Test case {i} has invalid quantization level: {test_case['quant_level']}")
                
            # Validate prompt type
            if test_case['prompt_type'] not in available_prompt_types:
                raise ValueError(f"Test case {i} has invalid prompt type: {test_case['prompt_type']}")
                
            # Validate image path exists
            # Strip leading 'data/' from the path if it exists
            image_path = test_case['image_path']
            if image_path.startswith('data/'):
                image_path = image_path[5:]  # Remove 'data/' prefix
            image_path = data_dir / image_path
            if not image_path.exists():
                raise ValueError(f"Test case {i} image path does not exist: {image_path}")
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating test matrix: {str(e)}")
        raise

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