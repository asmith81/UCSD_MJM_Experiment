"""
Llama Vision Model Evaluation Notebook

This notebook evaluates the Llama Vision model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Llama Vision Model Evaluation
# 
# This notebook evaluates the Llama Vision model's performance across different quantization levels
# and prompt strategies for invoice data extraction.

# %% [markdown]
# ## Environment Setup
# 
# First, we need to set up the environment and import required modules.

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging

# Determine root directory
try:
    # When running as a script
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    # When running in a notebook, look for project root markers
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
            ROOT_DIR = current_dir
            break
        current_dir = current_dir.parent
    else:
        raise RuntimeError("Could not find project root directory. Make sure you're running from within the project structure.")

sys.path.append(str(ROOT_DIR))

# Install dependencies
print("Installing dependencies...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")])
    print("Dependencies installed successfully.")
    
    # Install PyTorch dependencies separately
    print("Installing PyTorch dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    print("PyTorch dependencies installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error installing dependencies: {e}")
    raise

# Import project modules
from src import execution
from src.environment import setup_environment
from src.config import load_yaml_config
from src.models.llama_vision import load_model, process_image_wrapper, validate_results
from src.prompts import load_prompt_template
from src.results_logging import track_execution, log_result, ResultStructure

# Setup environment
try:
    env = setup_environment()
    paths = env['paths']
    
    # Validate paths
    required_paths = ['logs', 'models', 'data']
    missing_paths = [path for path in required_paths if path not in paths]
    if missing_paths:
        raise RuntimeError(f"Missing required paths in environment: {missing_paths}")
        
    # Ensure log directory exists
    paths['logs'].mkdir(parents=True, exist_ok=True)
    
except Exception as e:
    logger.error(f"Error setting up environment: {str(e)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = ROOT_DIR / "config" / "llama_vision.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

try:
    config = load_yaml_config(str(config_path))
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# %% [markdown]
# ## Model Configuration
# 
# Set the model name and test matrix path.

# %%
# Set model for this notebook
MODEL_NAME = "llama_vision"
TEST_MATRIX_PATH = str(ROOT_DIR / "config" / "test_matrix.json")
EXECUTION_LOG_PATH = paths['logs'] / f"{MODEL_NAME}_execution.log"

# Validate test matrix exists
if not Path(TEST_MATRIX_PATH).exists():
    raise FileNotFoundError(f"Test matrix file not found: {TEST_MATRIX_PATH}")

# Load model configuration
try:
    model_config = config['model']
    prompt_config = config['prompts']
except KeyError as e:
    logger.error(f"Missing required configuration section: {e}")
    raise

# %% [markdown]
# ## Run Test Suite
# 
# Execute the test suite for Llama Vision model.

# %%
def main():
    """Run the test suite and handle results."""
    try:
        # Track execution start
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "started"
        )
        
        # Run test suite
        results = execution.run_test_suite(
            model_name=MODEL_NAME,
            test_matrix_path=TEST_MATRIX_PATH,
            model_loader=load_model,
            processor=process_image_wrapper,
            prompt_loader=load_prompt_template,
            result_validator=validate_results
        )
        
        # Log results
        logger.info(f"Completed {len(results)} test cases for {MODEL_NAME}")
        
        # Track execution completion
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "completed"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error running test suite: {str(e)}")
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "failed",
            str(e)
        )
        raise

if __name__ == "__main__":
    results = main()

# %% [markdown]
# ## Results Analysis
# 
# Analyze and visualize the results for Llama Vision model.

# %%
def analyze_results(results: list) -> dict:
    """Analyze test results for Llama Vision model."""
    if not results:
        logger.warning("No results to analyze")
        return {}
        
    # Validate result structure
    required_fields = ['test_parameters', 'evaluation', 'model_response']
    for result in results:
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            logger.warning(f"Result missing required fields: {missing_fields}")
            continue
            
    # Group results by quantization level
    quant_results = {}
    for result in results:
        try:
            quant_level = result['test_parameters']['quantization']
            if quant_level not in quant_results:
                quant_results[quant_level] = []
            quant_results[quant_level].append(result)
        except KeyError:
            logger.warning("Result missing quantization level")
            continue
    
    # Calculate metrics per quantization level
    metrics = {}
    for quant_level, quant_results in quant_results.items():
        if not quant_results:
            continue
            
        total_tests = len(quant_results)
        correct_work_order = sum(1 for r in quant_results 
                               if r['evaluation']['work_order_number']['normalized_match'])
        correct_total_cost = sum(1 for r in quant_results 
                               if r['evaluation']['total_cost']['normalized_match'])
        
        metrics[quant_level] = {
            'work_order_accuracy': correct_work_order / total_tests,
            'total_cost_accuracy': correct_total_cost / total_tests,
            'avg_processing_time': sum(r['model_response']['processing_time'] 
                                    for r in quant_results) / total_tests
        }
    
    return metrics

# Analyze results
if 'results' in locals():
    metrics = analyze_results(results)
    if metrics:
        print("\nPerformance Metrics by Quantization Level:")
        for quant_level, metric in metrics.items():
            print(f"\n{quant_level}-bit Quantization:")
            print(f"Work Order Accuracy: {metric['work_order_accuracy']:.2%}")
            print(f"Total Cost Accuracy: {metric['total_cost_accuracy']:.2%}")
            print(f"Average Processing Time: {metric['avg_processing_time']:.2f}s")
    else:
        print("No valid results to analyze") 