"""
Tests for core utilities that can run in both local and remote environments.
"""

import pytest
from pathlib import Path
from src.environment import setup_paths, check_cuda_availability, get_environment_overrides
from src.data_utils import load_image, validate_ground_truth
from src.results_logging import log_result, load_result

def test_setup_paths(project_root: Path):
    """Test directory structure setup."""
    paths = setup_paths(project_root)
    
    # Check all required directories exist
    required_dirs = ['data_dir', 'models_dir', 'results_dir', 'logs_dir', 
                    'config_dir', 'prompts_dir']
    for dir_name in required_dirs:
        assert dir_name in paths
        assert paths[dir_name].exists()
        assert paths[dir_name].is_dir()

def test_environment_overrides(environment: str):
    """Test environment-specific overrides."""
    overrides = get_environment_overrides()
    
    if environment == "local":
        assert overrides['use_gpu'] is False
        assert overrides['log_level'] == 'DEBUG'
    else:
        assert overrides['use_gpu'] is True
        assert overrides['log_level'] == 'INFO'

def test_cuda_availability(cuda_available: bool, environment: str):
    """Test CUDA availability check."""
    cuda_info = check_cuda_availability()
    
    if environment == "local":
        # Local environment may or may not have CUDA
        assert isinstance(cuda_info['cuda_available'], bool)
    else:
        # Remote environment should have CUDA
        assert cuda_info['cuda_available'] is True
        assert cuda_info['cuda_version'] is not None

def test_mock_data_loading(mock_data_dir: Path):
    """Test loading mock data."""
    # Create mock data if it doesn't exist
    mock_image = mock_data_dir / "test_image.jpg"
    mock_gt = mock_data_dir / "test_ground_truth.json"
    
    if not mock_data_dir.exists():
        mock_data_dir.mkdir(parents=True)
    
    # Test image loading
    try:
        image = load_image(mock_image)
        assert image is not None
    except FileNotFoundError:
        pytest.skip("Mock image not found")
    
    # Test ground truth validation
    try:
        gt_data = validate_ground_truth(mock_gt)
        assert gt_data is not None
    except FileNotFoundError:
        pytest.skip("Mock ground truth not found")

def test_result_logging(project_root: Path):
    """Test result logging functionality."""
    test_result = {
        "model": "test_model",
        "quantization": 32,
        "prompt_strategy": "test",
        "raw_value": "TEST123",
        "normalized_value": "TEST123",
        "confidence": 0.95,
        "processing_time": 0.1
    }
    
    # Test logging
    result_path = project_root / "results" / "test_result.json"
    log_result(result_path, test_result)
    
    # Test loading
    loaded_result = load_result(result_path)
    assert loaded_result == test_result
    
    # Cleanup
    result_path.unlink() 