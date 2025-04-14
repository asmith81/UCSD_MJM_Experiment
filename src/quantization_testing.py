"""
Quantization testing utilities for model evaluation.

This module provides functions for running test suites across different quantization levels.
"""

import json
import tempfile
import os
import logging
from typing import Dict, List, Callable, Any
from pathlib import Path
from src import execution

logger = logging.getLogger(__name__)

def run_quantization_test_suite(
    model_name: str,
    test_matrix: Dict,
    model_loader: Callable,
    processor: Callable,
    prompt_loader: Callable,
    result_validator: Callable,
    project_root: Path,
    execution_log_path: Path
) -> Dict[int, List[Dict]]:
    """
    Run test suite across all quantization levels.
    
    Args:
        model_name: Name of the model to test
        test_matrix: Full test matrix dictionary
        model_loader: Function to load model with specific quantization
        processor: Function to process test cases
        prompt_loader: Function to load prompt templates
        result_validator: Function to validate results
        project_root: Path to project root directory
        execution_log_path: Path to execution log file
        
    Returns:
        Dictionary mapping quantization levels to their test results
    """
    results = {}
    
    # Get unique quantization levels from test matrix
    quant_levels = sorted(set(
        case['quant_level'] for case in test_matrix['test_cases']
        if case['model_name'] == model_name
    ))
    
    for quant_level in quant_levels:
        try:
            # Track execution start
            track_execution(
                execution_log_path,
                model_name,
                f"quant_{quant_level}",
                0,
                "started"
            )
            
            # Run test suite for this quantization level
            level_results = _run_quantization_level(
                quant_level=quant_level,
                model_name=model_name,
                test_matrix=test_matrix,
                model_loader=model_loader,
                processor=processor,
                prompt_loader=prompt_loader,
                result_validator=result_validator,
                project_root=project_root
            )
            
            results[quant_level] = level_results
            
            # Track execution completion
            track_execution(
                execution_log_path,
                model_name,
                f"quant_{quant_level}",
                0,
                "completed"
            )
            
        except Exception as e:
            logger.error(f"Error running test suite for {quant_level}-bit quantization: {str(e)}")
            track_execution(
                execution_log_path,
                model_name,
                f"quant_{quant_level}",
                0,
                "failed",
                str(e)
            )
            raise
    
    return results

def _run_quantization_level(
    quant_level: int,
    model_name: str,
    test_matrix: Dict,
    model_loader: Callable,
    processor: Callable,
    prompt_loader: Callable,
    result_validator: Callable,
    project_root: Path
) -> List[Dict]:
    """
    Run test suite for a specific quantization level.
    
    Args:
        quant_level: Quantization level to test
        model_name: Name of the model to test
        test_matrix: Full test matrix dictionary
        model_loader: Function to load model with specific quantization
        processor: Function to process test cases
        prompt_loader: Function to load prompt templates
        result_validator: Function to validate results
        project_root: Path to project root directory
        
    Returns:
        List of test results for this quantization level
    """
    # Filter test cases for this quantization level and model
    quant_test_cases = [
        case for case in test_matrix['test_cases']
        if case['quant_level'] == quant_level and case['model_name'] == model_name
    ]
    
    if not quant_test_cases:
        logger.warning(f"No test cases found for {model_name} at {quant_level}-bit quantization")
        return []
    
    # Create temporary test matrix file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump({'test_cases': quant_test_cases}, temp_file)
        temp_path = temp_file.name
    
    try:
        # Run test suite for this quantization level
        results = execution.run_test_suite(
            model_name=model_name,
            test_matrix_path=temp_path,
            model_loader=model_loader,
            processor=processor,
            prompt_loader=prompt_loader,
            result_validator=result_validator,
            project_root=project_root
        )
        
        # Log results
        logger.info(f"Completed {len(results)} test cases for {model_name} at {quant_level}-bit quantization")
        return results
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_path}: {e}") 