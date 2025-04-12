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
    def __call__(self, model: Any, prompt_template: str) -> Dict[str, Any]:
        ...

def load_test_matrix(test_matrix_path: str) -> pd.DataFrame:
    """
    Load and validate the test matrix CSV file.
    
    Args:
        test_matrix_path: Path to the test matrix CSV file
        
    Returns:
        DataFrame containing test cases
        
    Raises:
        FileNotFoundError: If test matrix file doesn't exist
        ValueError: If test matrix is invalid
    """
    if not Path(test_matrix_path).exists():
        raise FileNotFoundError(f"Test matrix file not found: {test_matrix_path}")
        
    df = pd.read_csv(test_matrix_path)
    
    # Validate required columns
    required_columns = ['model', 'quantization', 'prompt_strategy']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Test matrix missing required columns: {missing_columns}")
        
    # Validate quantization values
    valid_quantization = [4, 8, 16, 32]
    invalid_quantization = df[~df['quantization'].isin(valid_quantization)]['quantization'].unique()
    if len(invalid_quantization) > 0:
        raise ValueError(f"Invalid quantization values found: {invalid_quantization}")
        
    return df

def run_test_suite(
    model_name: str, 
    test_matrix_path: str,
    model_loader: Optional[ModelLoader] = None,
    processor: Optional[ModelProcessor] = None,
    prompt_loader: Optional[Callable[[str], str]] = None,
    result_validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
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
    model_cases = test_cases[test_cases['model'] == model_name]
    
    if len(model_cases) == 0:
        raise ValueError(f"No test cases found for model: {model_name}")
        
    results = []
    
    # Group by quantization to minimize model reloads
    for quantization in sorted(model_cases['quantization'].unique()):
        logger.info(f"Loading model {model_name} with quantization {quantization}")
        
        try:
            # Load model using provided loader
            model = model_loader(model_name, quantization)
            
            # Get all prompt strategies for this quantization
            quantization_cases = model_cases[model_cases['quantization'] == quantization]
            
            for _, case in quantization_cases.iterrows():
                logger.info(f"Running test case: {case['prompt_strategy']}")
                try:
                    # Load prompt template
                    prompt_template = prompt_loader(case['prompt_strategy'])
                    
                    # Run inference
                    result = processor(model, prompt_template)
                    
                    # Validate results
                    validated_result = result_validator(result)
                    
                    # Add test case metadata
                    validated_result.update({
                        'model': model_name,
                        'quantization': quantization,
                        'prompt_strategy': case['prompt_strategy']
                    })
                    
                    results.append(validated_result)
                    
                except Exception as e:
                    logger.error(f"Error in test case {case['prompt_strategy']}: {str(e)}")
                    results.append({
                        'model': model_name,
                        'quantization': quantization,
                        'prompt_strategy': case['prompt_strategy'],
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