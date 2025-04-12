"""
Tests for model-specific functionality that adapts to the environment.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from src.models import load_model, run_inference

@pytest.mark.skipif(
    "environment == 'local'",
    reason="Model loading tests require GPU environment"
)
def test_model_loading(project_root: Path, test_config: Dict[str, Any]):
    """Test model loading in GPU environment."""
    for model_name in ["pixtral", "llama_vision", "doctr"]:
        for quantization in test_config["quantization_levels"]:
            try:
                model = load_model(model_name, quantization)
                assert model is not None
                assert hasattr(model, "name")
                assert model.name == model_name
                assert hasattr(model, "quantization")
                assert model.quantization == quantization
            except Exception as e:
                pytest.fail(f"Failed to load {model_name} with quantization {quantization}: {str(e)}")

def test_mock_inference(mock_model, test_config: Dict[str, Any]):
    """Test inference with mock model in local environment."""
    test_prompt = "Extract the work order number from this invoice."
    
    result = run_inference(mock_model, test_prompt)
    
    assert isinstance(result, dict)
    assert "raw_value" in result
    assert "normalized_value" in result
    assert "confidence" in result
    assert "processing_time" in result
    
    assert result["raw_value"] == "MOCK123"
    assert result["normalized_value"] == "MOCK123"
    assert 0 <= result["confidence"] <= 1
    assert result["processing_time"] > 0

@pytest.mark.skipif(
    "environment == 'local'",
    reason="Full model inference tests require GPU environment"
)
def test_full_model_inference(project_root: Path, test_config: Dict[str, Any]):
    """Test full model inference in GPU environment."""
    model = load_model("pixtral", 32)  # Use full precision for testing
    test_prompt = "Extract the work order number from this invoice."
    
    result = run_inference(model, test_prompt)
    
    assert isinstance(result, dict)
    assert "raw_value" in result
    assert "normalized_value" in result
    assert "confidence" in result
    assert "processing_time" in result
    
    assert isinstance(result["raw_value"], str)
    assert isinstance(result["normalized_value"], str)
    assert 0 <= result["confidence"] <= 1
    assert result["processing_time"] > 0

def test_prompt_strategies(test_config: Dict[str, Any]):
    """Test different prompt strategies."""
    for strategy in test_config["prompt_strategies"]:
        # Test that prompt templates exist and are valid
        prompt_path = Path(f"config/prompts/{strategy}.txt")
        assert prompt_path.exists()
        
        with open(prompt_path, "r") as f:
            prompt = f.read()
            assert len(prompt) > 0
            assert "{field}" in prompt  # Check for field placeholder 