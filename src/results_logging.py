"""
Result tracking and logging for LMM invoice data extraction comparison.
Handles result capture, execution tracking, and result structure definitions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
from .data_utils import normalize_work_order, normalize_total_cost

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

def create_result_structure(
    model_name: str,
    prompt_type: str,
    quant_level: int,
    environment: str = "RunPod T4 GPU"
) -> Dict[str, Any]:
    """
    Create base result structure for a test run.
    
    Args:
        model_name: Name of the model being tested
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        environment: Testing environment description
        
    Returns:
        Base result structure dictionary
    """
    return {
        "meta": {
            "experiment_id": f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "environment": environment
        },
        "test_parameters": {
            "model_name": model_name,
            "field_type": "both",
            "prompt_type": prompt_type,
            "quant_level": quant_level
        },
        "results_by_image": {}
    }

def validate_model_output(model_output: Dict[str, Any]) -> bool:
    """
    Validate model output structure.
    
    Args:
        model_output: Model output to validate
        
    Returns:
        True if structure is valid
        
    Raises:
        ValueError: If structure is invalid
    """
    required_fields = ['work_order_number', 'total_cost']
    required_subfields = ['raw_text', 'parsed_value', 'normalized_value']
    
    for field in required_fields:
        if field not in model_output:
            raise ValueError(f"Missing required field: {field}")
            
        for subfield in required_subfields:
            if subfield not in model_output[field]:
                raise ValueError(f"Missing required subfield {subfield} in {field}")
                
    return True

def log_result(
    result_path: Union[str, Path],
    image_id: str,
    model_output: Dict[str, Any],
    ground_truth: Dict[str, Any],
    processing_time: float,
    model_name: str,
    prompt_type: str,
    quant_level: int,
    environment: str = "RunPod T4 GPU"
) -> None:
    """
    Log result for a single image.
    
    Args:
        result_path: Path to result file
        image_id: ID of the image
        model_output: Model output for the image
        ground_truth: Ground truth for the image
        processing_time: Time taken for processing
        model_name: Name of the model being tested
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        environment: Testing environment description
        
    Raises:
        ValueError: If model output is invalid
    """
    try:
        # Validate model output
        validate_model_output(model_output)
        
        # Create result entry with both fields
        result_entry = {
            "ground_truth": ground_truth,
            "model_response": {
                "work_order_number": {
                    "raw_text": model_output['work_order_number']['raw_text'],
                    "parsed_value": model_output['work_order_number']['parsed_value'],
                    "normalized_value": model_output['work_order_number']['normalized_value']
                },
                "total_cost": {
                    "raw_text": model_output['total_cost']['raw_text'],
                    "parsed_value": model_output['total_cost']['parsed_value'],
                    "normalized_value": model_output['total_cost']['normalized_value']
                },
                "processing_time": processing_time
            },
            "evaluation": {
                "work_order_number": {
                    "raw_string_match": (
                        model_output['work_order_number']['parsed_value'] == 
                        ground_truth['work_order_number']['raw_value']
                    ),
                    "normalized_match": (
                        model_output['work_order_number']['normalized_value'] == 
                        ground_truth['work_order_number']['normalized_value']
                    ),
                    "cer": calculate_cer(
                        model_output['work_order_number']['parsed_value'],
                        ground_truth['work_order_number']['raw_value']
                    ),
                    "error_category": categorize_error(
                        model_output['work_order_number']['parsed_value'],
                        ground_truth['work_order_number']['raw_value'],
                        "work_order"
                    )
                },
                "total_cost": {
                    "raw_string_match": (
                        model_output['total_cost']['parsed_value'] == 
                        ground_truth['total_cost']['raw_value']
                    ),
                    "normalized_match": (
                        model_output['total_cost']['normalized_value'] == 
                        ground_truth['total_cost']['normalized_value']
                    ),
                    "cer": calculate_cer(
                        model_output['total_cost']['parsed_value'],
                        ground_truth['total_cost']['raw_value']
                    ),
                    "error_category": categorize_error(
                        model_output['total_cost']['parsed_value'],
                        ground_truth['total_cost']['raw_value'],
                        "total_cost"
                    )
                }
            }
        }
        
        # Append to result file
        result_path = Path(result_path)
        if result_path.exists():
            with open(result_path, 'r') as f:
                results = json.load(f)
        else:
            results = create_result_structure(
                model_name=model_name,
                prompt_type=prompt_type,
                quant_level=quant_level,
                environment=environment
            )
            
        results['results_by_image'][image_id] = result_entry
        
        # Write updated results
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error logging result for image {image_id}: {str(e)}")
        raise

def track_execution(
    execution_log_path: Union[str, Path],
    model_name: str,
    prompt_type: str,
    quant_level: int,
    status: str,
    error: Optional[str] = None
) -> None:
    """
    Track execution status for a test run.
    
    Args:
        execution_log_path: Path to execution log file
        model_name: Name of the model
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        status: Execution status
        error: Error message if any
    """
    try:
        execution_log_path = Path(execution_log_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt_type": prompt_type,
            "quant_level": quant_level,
            "status": status,
            "error": error
        }
        
        # Append to execution log
        if execution_log_path.exists():
            df = pd.read_csv(execution_log_path)
        else:
            df = pd.DataFrame(columns=[
                'timestamp', 'model_name', 'prompt_type',
                'quant_level', 'status', 'error'
            ])
            
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(execution_log_path, index=False)
        
    except Exception as e:
        logger.error(f"Error tracking execution: {str(e)}")
        raise 