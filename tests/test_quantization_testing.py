"""
Tests for quantization testing functionality.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock
from src.quantization_testing import _run_quantization_level

@pytest.fixture
def mock_test_matrix() -> Dict:
    """Create a mock test matrix for testing."""
    return {
        'test_cases': [
            {
                'model_name': 'test_model',
                'quant_level': 4,
                'image_path': 'test_image.jpg',
                'field_type': 'invoice_number',
                'prompt_type': 'basic',
                'expected_value': '12345'
            },
            {
                'model_name': 'test_model',
                'quant_level': 8,
                'image_path': 'test_image.jpg',
                'field_type': 'invoice_number',
                'prompt_type': 'basic',
                'expected_value': '12345'
            }
        ]
    }

@pytest.fixture
def mock_model_loader():
    """Create a mock model loader function."""
    def loader(model_name: str, quant_level: int):
        return f"mock_model_{model_name}_{quant_level}bit"
    return loader

@pytest.fixture
def mock_processor():
    """Create a mock processor function."""
    def processor(model, prompt, test_case):
        return {
            'model': model,
            'prompt': prompt,
            'test_case': test_case,
            'result': '12345'
        }
    return processor

@pytest.fixture
def mock_prompt_loader():
    """Create a mock prompt loader function."""
    def loader(strategy: str):
        return f"mock_prompt_{strategy}"
    return loader

@pytest.fixture
def mock_result_validator():
    """Create a mock result validator function."""
    def validator(result, expected_value):
        return {
            'is_correct': result['result'] == expected_value,
            'actual_value': result['result'],
            'expected_value': expected_value
        }
    return validator

@pytest.fixture
def mock_execution_results():
    """Create mock results from execution module."""
    return [{
        'model': 'mock_model_test_model_4bit',
        'prompt': 'mock_prompt_basic',
        'test_case': {
            'model_name': 'test_model',
            'quant_level': 4,
            'image_path': 'test_image.jpg',
            'field_type': 'invoice_number',
            'prompt_type': 'basic',
            'expected_value': '12345'
        },
        'result': '12345',
        'is_correct': True,
        'actual_value': '12345',
        'expected_value': '12345'
    }]

def test_run_quantization_level(
    mock_test_matrix: Dict,
    mock_model_loader,
    mock_processor,
    mock_prompt_loader,
    mock_result_validator,
    mock_execution_results,
    tmp_path: Path
):
    """
    Test that _run_quantization_level correctly processes test cases
    and returns results for a specific quantization level.
    """
    # Mock the execution module's run_test_suite function
    with patch('src.execution.run_test_suite', return_value=mock_execution_results):
        # Run the test suite for 4-bit quantization
        results = _run_quantization_level(
            quant_level=4,
            model_name='test_model',
            test_matrix=mock_test_matrix,
            model_loader=mock_model_loader,
            processor=mock_processor,
            prompt_loader=mock_prompt_loader,
            result_validator=mock_result_validator,
            project_root=tmp_path
        )
        
        # Verify results
        assert len(results) == 1  # Should have one result for 4-bit quantization
        result = results[0]
        
        # Verify the result structure
        assert 'model' in result
        assert 'prompt' in result
        assert 'test_case' in result
        assert 'result' in result
        assert 'is_correct' in result
        assert 'actual_value' in result
        assert 'expected_value' in result
        
        # Verify specific values
        assert result['model'] == 'mock_model_test_model_4bit'
        assert result['prompt'] == 'mock_prompt_basic'
        assert result['test_case']['quant_level'] == 4
        assert result['is_correct'] is True
        assert result['actual_value'] == '12345'
        assert result['expected_value'] == '12345'

def test_run_quantization_level_no_cases(
    mock_test_matrix: Dict,
    mock_model_loader,
    mock_processor,
    mock_prompt_loader,
    mock_result_validator,
    tmp_path: Path
):
    """
    Test that _run_quantization_level returns empty list when no test cases
    are found for the specified quantization level.
    """
    # Run the test suite for a non-existent quantization level
    results = _run_quantization_level(
        quant_level=16,  # No test cases for 16-bit in our mock data
        model_name='test_model',
        test_matrix=mock_test_matrix,
        model_loader=mock_model_loader,
        processor=mock_processor,
        prompt_loader=mock_prompt_loader,
        result_validator=mock_result_validator,
        project_root=tmp_path
    )
    
    # Verify empty results
    assert len(results) == 0 