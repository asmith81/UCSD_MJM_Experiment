"""
Test script for verifying image path handling and model processing.
"""

import sys
from pathlib import Path
import json
import logging
from PIL import Image

# Get project root and add to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pixtral import load_model, process_image_wrapper
from src.data_utils import DataConfig, setup_data_paths
from src.environment import setup_environment
from src.config import load_yaml_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_image_processing():
    """Test image path handling and model processing."""
    try:
        # Setup environment
        env = setup_environment(
            project_root=project_root,
            requirements_path=project_root / "requirements.txt"
        )
        
        # Load config
        config_path = project_root / "config" / "models" / "pixtral.yaml"
        config = load_yaml_config(str(config_path))
        
        # Setup data config
        data_config = setup_data_paths(
            env_config=env,
            image_extensions=['.jpg', '.jpeg', '.png'],
            max_image_size=1120,
            supported_formats=['RGB', 'L']
        )
        
        # Load test matrix
        test_matrix_path = project_root / "config" / "test_matrix.json"
        with open(test_matrix_path, 'r') as f:
            test_matrix = json.load(f)
        
        # Get first test case for pixtral model
        test_case = next(
            case for case in test_matrix['test_cases']
            if case['model_name'] == 'pixtral'
        )
        
        print(f"\nTesting with image: {test_case['image_path']}")
        
        # Verify image exists
        image_path = Path(test_case['image_path'])
        if not image_path.is_absolute():
            image_path = env['data_dir'] / image_path
            
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        print(f"Image found at: {image_path}")
        
        # Load model
        model = load_model(
            model_name='pixtral',
            quantization=test_case['quant_level'],
            models_dir=env['models_dir'],
            config=config
        )
        
        # Process image
        result = process_image_wrapper(
            model=model,
            prompt_template="Please extract the work order number and total cost from this invoice.",
            image_path=str(image_path),
            field_type=test_case['field_type'],
            config=data_config
        )
        
        print("\nModel Response:")
        print(f"Output: {result['model_response']['output']}")
        print(f"Processing Time: {result['model_response']['processing_time']:.2f}s")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_image_processing() 