"""
Test execution framework for model evaluation.

This module provides the core functionality for running test suites across different
models, quantization levels, and prompt strategies. It follows a functional approach
with stateless functions that transform data.
"""

import pandas as pd
from typing import Dict, Any, List, Callable, Optional, Protocol
import logging
from pathlib import Path
import json
from .results_logging import evaluate_model_output
from .data_utils import GroundTruthData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelLoader(Protocol):
    """Protocol for model loading functions."""
    def __call__(self, model_name: str, quantization: int) -> Any:
        ...

class ModelProcessor(Protocol):
    """Protocol for model processing functions."""
    def __call__(self, model: Any, prompt_template: str, case: Dict[str, Any]) -> Dict[str, Any]:
        ...

def load_test_matrix(test_matrix_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate the test matrix JSON file.
    
    Args:
        test_matrix_path: Path to the test matrix JSON file
        
    Returns:
        List of test cases
        
    Raises:
        FileNotFoundError: If test matrix file doesn't exist
        ValueError: If test matrix is invalid
    """
    if not Path(test_matrix_path).exists():
        raise FileNotFoundError(f"Test matrix file not found: {test_matrix_path}")
        
    try:
        with open(test_matrix_path, 'r') as f:
            data = json.load(f)
            
        if 'test_cases' not in data:
            raise ValueError("Test matrix must contain 'test_cases' array")
            
        # Validate required fields
        required_fields = ['model_name', 'field_type', 'prompt_type', 'quant_level']
        for case in data['test_cases']:
            missing_fields = [field for field in required_fields if field not in case]
            if missing_fields:
                raise ValueError(f"Test case missing required fields: {missing_fields}")
                
        # Validate quantization values
        valid_quantization = [4, 8, 16, 32]
        invalid_quantization = [case['quant_level'] for case in data['test_cases'] 
                              if case['quant_level'] not in valid_quantization]
        if invalid_quantization:
            raise ValueError(f"Invalid quantization values found: {invalid_quantization}")
            
        return data['test_cases']
        
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in test matrix file: {test_matrix_path}")

def run_test_suite(
    model_name: str, 
    test_matrix_path: str,
    model_loader: Optional[ModelLoader] = None,
    processor: Optional[ModelProcessor] = None,
    prompt_loader: Optional[Callable[[str], str]] = None,
    result_validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    project_root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Run the test suite for a specific model.
    
    Args:
        model_name: Name of the model to test
        test_matrix_path: Path to the test matrix CSV file
        model_loader: Optional function to load the model
        processor: Optional function to process images
        prompt_loader: Optional function to load prompt templates
        result_validator: Optional function to validate results
        project_root: Optional path to project root directory
        
    Returns:
        List of test results
        
    Raises:
        ValueError: If model_name is not found in test matrix
        RuntimeError: If required dependencies are not provided
    """
    # Validate dependencies
    if not all([model_loader, processor, prompt_loader, result_validator]):
        raise RuntimeError("All dependencies (model_loader, processor, prompt_loader, result_validator) must be provided")
    
    # Load and filter test matrix
    test_cases = load_test_matrix(test_matrix_path)
    model_cases = [case for case in test_cases if case['model_name'] == model_name]
    
    if len(model_cases) == 0:
        raise ValueError(f"No test cases found for model: {model_name}")
        
    results = []
    
    # Group by quantization to minimize model reloads
    for quantization in sorted(set(case['quant_level'] for case in model_cases)):
        logger.info(f"Loading model {model_name} with quantization {quantization}")
        
        try:
            # Load model using provided loader
            model = model_loader(model_name, quantization)
            
            # Get all prompt strategies for this quantization
            quantization_cases = [case for case in model_cases if case['quant_level'] == quantization]
            
            for case in quantization_cases:
                logger.info(f"Running test case: {case['prompt_type']}")
                try:
                    # Load prompt template
                    prompt_template = prompt_loader(case['prompt_type'])
                    
                    # Convert relative image path to absolute if project_root is provided
                    image_path = case['image_path']
                    if project_root is not None:
                        image_path = str(project_root / image_path)
                    
                    # Run inference
                    result = processor(model, prompt_template, case)
                    
                    # Get ground truth
                    ground_truth = case.get('ground_truth', {})
                    
                    # Evaluate results
                    evaluation = evaluate_model_output(
                        result['model_response']['output'],
                        ground_truth,
                        case['field_type']
                    )
                    
                    # Add evaluation to result
                    result['evaluation'] = evaluation
                    
                    # Validate results
                    validated_result = result_validator(result)
                    
                    # Add test case metadata
                    validated_result.update({
                        'model': model_name,
                        'quantization': quantization,
                        'prompt_type': case['prompt_type']
                    })
                    
                    results.append(validated_result)
                    
                except Exception as e:
                    logger.error(f"Error in test case {case['prompt_type']}: {str(e)}")
                    results.append({
                        'model': model_name,
                        'quantization': quantization,
                        'prompt_type': case['prompt_type'],
                        'error': str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Error loading model {model_name} with quantization {quantization}: {str(e)}")
            results.append({
                'model': model_name,
                'quantization': quantization,
                'error': str(e)
            })
                
    return results

def load_model(model_name: str, quantization: int):
    """
    Load a model with specific quantization.
    
    Args:
        model_name: Name of the model to load
        quantization: Quantization level to use
        
    Returns:
        Loaded model instance
        
    Raises:
        NotImplementedError: If model loading is not implemented
    """
    # This will be implemented in the model-specific modules
    raise NotImplementedError("Model loading not implemented")

def load_prompt_template(prompt_strategy: str) -> str:
    """
    Load a prompt template for a specific strategy.
    
    Args:
        prompt_strategy: Name of the prompt strategy
        
    Returns:
        Loaded prompt template
        
    Raises:
        FileNotFoundError: If prompt template doesn't exist
    """
    # This will be implemented in the prompts module
    raise NotImplementedError("Prompt template loading not implemented")

def run_inference(model, prompt_template: str) -> Dict[str, Any]:
    """
    Run inference with the model and prompt template.
    
    Args:
        model: Loaded model instance
        prompt_template: Prompt template to use
        
    Returns:
        Inference results
        
    Raises:
        NotImplementedError: If inference is not implemented
    """
    # This will be implemented in the model-specific modules
    raise NotImplementedError("Inference not implemented")

def validate_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize inference results.
    
    Args:
        results: Raw inference results
        
    Returns:
        Validated and normalized results
        
    Raises:
        NotImplementedError: If validation is not implemented
    """
    # This will be implemented in the evaluation module
    raise NotImplementedError("Result validation not implemented") 