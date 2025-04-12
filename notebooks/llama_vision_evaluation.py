"""
Llama Vision Model Evaluation Notebook

This notebook evaluates the Llama-3.2-11B-Vision model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Llama Vision Model Evaluation
# 
# This notebook evaluates the Llama-3.2-11B-Vision model's performance across different quantization levels
# and prompt strategies for invoice data extraction.

# %% [markdown]
# ## Environment Setup
# 
# First, we need to set up the environment and import required modules.

# %%
import os
import sys
from pathlib import Path
import logging

# Determine root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import project modules
import execution
from src.environment import setup_environment
from src.config import load_yaml_config
from src.models.llama_vision import load_model, process_image
from src.results_logging import track_execution, log_result, ResultStructure

# Setup environment
env = setup_environment()
paths = env['paths']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    config = load_yaml_config('config/llama_vision.yaml')
except FileNotFoundError:
    logger.error("Llama Vision configuration file not found")
    raise

# %% [markdown]
# ## Model Configuration
# 
# Set the model name and test matrix path.

# %%
# Set model for this notebook
MODEL_NAME = "llama_vision"
TEST_MATRIX_PATH = "config/test_matrix.csv"
EXECUTION_LOG_PATH = paths['logs'] / f"{MODEL_NAME}_execution.log"

# Load model configuration
model_config = config['model']
prompt_config = config['prompts']

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
            processor=process_image
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
    # Group results by quantization level
    quant_results = {}
    for result in results:
        quant_level = result['test_parameters']['quantization']
        if quant_level not in quant_results:
            quant_results[quant_level] = []
        quant_results[quant_level].append(result)
    
    # Calculate metrics per quantization level
    metrics = {}
    for quant_level, quant_results in quant_results.items():
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
    print("\nPerformance Metrics by Quantization Level:")
    for quant_level, metric in metrics.items():
        print(f"\n{quant_level}-bit Quantization:")
        print(f"Work Order Accuracy: {metric['work_order_accuracy']:.2%}")
        print(f"Total Cost Accuracy: {metric['total_cost_accuracy']:.2%}")
        print(f"Average Processing Time: {metric['avg_processing_time']:.2f}s") 