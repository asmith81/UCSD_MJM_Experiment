"""
Result validation module.

This module provides functionality for validating test results across different models.
"""

from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        required_fields = ['test_parameters', 'model_response', 'evaluation']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Result missing required fields: {missing_fields}")
            
        # Validate test parameters
        if 'model' not in result['test_parameters']:
            raise ValueError("Missing model name in test parameters")
        if 'quantization' not in result['test_parameters']:
            raise ValueError("Missing quantization level in test parameters")
            
        # Validate model response - allow both direct and chat-style formats
        if 'output' not in result['model_response']:
            # Try to find output in chat-style format
            if 'parsed_value' in result['model_response']:
                result['model_response']['output'] = result['model_response']['parsed_value']
            else:
                raise ValueError("Missing output in model response")
                
        if 'processing_time' not in result['model_response']:
            raise ValueError("Missing processing time in model response")
            
        # Validate evaluation
        if 'work_order_number' not in result['evaluation']:
            raise ValueError("Missing work order number evaluation")
        if 'total_cost' not in result['evaluation']:
            raise ValueError("Missing total cost evaluation")
            
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