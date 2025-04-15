"""
Pixtral Model Evaluation Notebook

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Pixtral Model Evaluation
# 
# This notebook evaluates the Pixtral-12B model's performance across different quantization levels
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
import json
from typing import Dict, Any, List

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    print("PyTorch dependencies installed successfully.")
    
    # Install Flash Attention 2
    print("Installing Flash Attention 2...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "flash-attn==2.5.0",
        "--no-build-isolation"
    ])
    print("Flash Attention 2 installed successfully.")
    
except subprocess.CalledProcessError as e:
    logger.error(f"Error installing dependencies: {e}")
    raise

# Import project modules
from src import execution
from src.environment import setup_environment, download_model
from src.config import load_yaml_config
from src.models.pixtral import load_model, process_image_wrapper, download_pixtral_model
from src.prompts import load_prompt_template
from src.results_logging import track_execution, log_result, ResultStructure, evaluate_model_output, validate_ground_truth, normalize_total_cost
from src.validation import validate_results, validate_test_matrix
from src.data_utils import DataConfig, setup_data_paths

# Setup environment
try:
    env = setup_environment(
        project_root=ROOT_DIR,
        requirements_path=ROOT_DIR / "requirements.txt"
    )
    
    # Validate paths
    required_paths = ['data_dir', 'models_dir', 'logs_dir', 'results_dir', 'prompts_dir']
    missing_paths = [path for path in required_paths if path not in env]
    if missing_paths:
        raise RuntimeError(f"Missing required paths in environment: {missing_paths}")
        
    # Ensure required directories exist
    for path in required_paths:
        env[path].mkdir(parents=True, exist_ok=True)
    
except Exception as e:
    logger.error(f"Error setting up environment: {str(e)}")
    raise

# Load configuration
config_path = ROOT_DIR / "config" / "models" / "pixtral.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

try:
    config = load_yaml_config(str(config_path))
    # Validate required configuration sections
    required_sections = ['name', 'loading', 'quantization', 'prompt', 'inference']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing required sections: {missing_sections}")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# Setup data configuration
try:
    data_config = setup_data_paths(
        env_config=env,
        image_extensions=['.jpg', '.jpeg', '.png'],
        max_image_size=1120,
        supported_formats=['RGB', 'L']
    )
    logger.info("Data configuration setup successfully")
except Exception as e:
    logger.error(f"Error setting up data configuration: {str(e)}")
    raise

# %% [markdown]
# ## Model Configuration
# 
# Set the model name and test matrix path.

# %%
# Set model for this notebook
MODEL_NAME = "pixtral"
TEST_MATRIX_PATH = str(ROOT_DIR / "config" / "test_matrix.json")
EXECUTION_LOG_PATH = env['logs_dir'] / f"{MODEL_NAME}_execution.log"

# Validate test matrix exists and is valid
try:
    if not Path(TEST_MATRIX_PATH).exists():
        raise FileNotFoundError(f"Test matrix file not found: {TEST_MATRIX_PATH}")
        
    # Load and validate test matrix
    with open(TEST_MATRIX_PATH, 'r') as f:
        test_matrix = json.load(f)
        
    # Validate test matrix structure
    validate_test_matrix(
        test_matrix=test_matrix,
        supported_quant_levels=[4, 8, 16, 32],
        available_prompt_types=['basic_extraction', 'detailed', 'few_shot', 'locational', 'step_by_step'],
        data_dir=env['data_dir']
    )
            
except Exception as e:
    logger.error(f"Error validating test matrix: {str(e)}")
    raise

# Load model configuration
try:
    # The config is already loaded and validated with required sections
    # We can use the config directly as it matches our needs
    model_config = {
        'name': config['name'],
        'path': config['repo_id'],
        'quantization_levels': list(config['quantization']['options'].keys())
    }
    
    prompt_config = {
        'format': config['prompt']['format'],
        'image_placeholder': config['prompt']['image_placeholder'],
        'default_field': config['prompt']['default_field']
    }
    
    # Validate model configuration
    required_model_fields = ['name', 'path', 'quantization_levels']
    missing_fields = [field for field in required_model_fields if field not in model_config]
    if missing_fields:
        raise ValueError(f"Model configuration missing required fields: {missing_fields}")
        
except KeyError as e:
    logger.error(f"Missing required configuration section: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model configuration: {str(e)}")
    raise

print(f"✓ Model configuration loaded successfully for {MODEL_NAME}")

# %% [markdown]
# ## Model Download
# 
# Download the model if it doesn't exist locally.

# %%
from src.models.pixtral import download_pixtral_model

# Set up model path
model_path = env['models_dir'] / "pixtral-12b"

# Download model if needed
try:
    if not model_path.exists():
        print(f"Downloading {MODEL_NAME} model...")
        if not download_pixtral_model(model_path, config['repo_id']):
            raise RuntimeError(f"Failed to download {MODEL_NAME} model")
        print(f"✓ {MODEL_NAME} model downloaded successfully")
    else:
        print(f"✓ {MODEL_NAME} model already exists at {model_path}")
except Exception as e:
    logger.error(f"Error downloading model: {str(e)}")
    raise

# %% [markdown]
# ## Single Test Validation
# 
# Test the entire pipeline with a single example to verify system functionality.

# %%
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def run_single_test():
    """Run a single test case through the entire pipeline."""
    try:
        # 1. Load first image
        image_dir = env['data_dir'] / 'images'
        image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        if not image_files:
            raise FileNotFoundError("No images found in data directory")
            
        first_image_path = image_files[0]
        print(f"\nLoading image: {first_image_path}")
        
        # Display image
        image = Image.open(first_image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
        # 2. Load ground truth
        ground_truth_path = env['data_dir'] / 'ground_truth.csv'
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Validate required columns
        required_columns = ['Invoice', 'Work Order Number/Numero de Orden', 'Total']
        missing_columns = [col for col in required_columns if col not in ground_truth_df.columns]
        if missing_columns:
            raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
        
        # Get ground truth for first image
        first_image_id = first_image_path.stem
        first_ground_truth = ground_truth_df[ground_truth_df['Invoice'] == int(first_image_id)].iloc[0]
        
        print("\nGround Truth Data:")
        display(Markdown(f"""
        - Invoice: {first_ground_truth['Invoice']}
        - Work Order Number: {first_ground_truth['Work Order Number/Numero de Orden']}
        - Total: {first_ground_truth['Total']}
        """))
        
        # 3. Load and display prompt
        prompt_strategy = "basic_extraction"
        prompt_template = load_prompt_template(
            prompt_strategy=prompt_strategy,
            prompts_dir=ROOT_DIR / "config" / "prompts"
        )
        
        print("\nGenerated Prompt:")
        display(Markdown(f"```\n{prompt_template}\n```"))
        
        # 4. Load model with default quantization
        model = load_model(
            model_name=MODEL_NAME,
            quantization=32,  # Using full precision for test
            models_dir=env['models_dir'],
            config=config
        )
        
        # 5. Process image and get model response
        print("\nRunning model inference...")
        result = process_image_wrapper(
            model=model,
            prompt_template=prompt_template,
            image_path=str(first_image_path),
            field_type="both",  # Testing both work order and total cost extraction
            config=data_config
        )
        
        # 6. Evaluate results using the new evaluation logic
        ground_truth = {
            'work_order_number': first_ground_truth['Work Order Number/Numero de Orden'],
            'total_cost': first_ground_truth['Total']
        }
        
        evaluation = evaluate_model_output(
            result['model_response']['output'],
            ground_truth,
            "both"
        )
        
        result['evaluation'] = evaluation
        
        print("\nModel Response:")
        display(Markdown(f"""
        - Raw Output: {result['model_response']['output']}
        - Processing Time: {result['model_response']['processing_time']:.2f}s
        - Evaluation:
            - Work Order Match: {result['evaluation']['work_order_number']['normalized_match']}
            - Work Order CER: {result['evaluation']['work_order_number']['cer']:.2f}
            - Total Cost Match: {result['evaluation']['total_cost']['normalized_match']}
            - Total Cost CER: {result['evaluation']['total_cost']['cer']:.2f}
        """))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in single test validation: {str(e)}")
        raise

# Run the test
test_result = run_single_test()

# %% [markdown]
# ## Run Test Suite by Quantization Level
# 
# Execute the test suite for Pixtral model, broken down by quantization level.

# %%
def run_quantization_level(quant_level: int, test_matrix: dict) -> list:
    """
    Run test suite for a specific quantization level.
    
    Args:
        quant_level: Quantization level to test (4, 8, 16, 32)
        test_matrix: Full test matrix dictionary
        
    Returns:
        List of test results
    """
    # Filter test cases for this quantization level and model
    quant_test_cases = [
        case for case in test_matrix['test_cases']
        if case['quant_level'] == quant_level and case['model_name'] == MODEL_NAME
    ]
    
    if not quant_test_cases:
        logger.warning(f"No test cases found for {MODEL_NAME} at {quant_level}-bit quantization")
        return []
    
    print(f"\nRunning {len(quant_test_cases)} test cases for {quant_level}-bit quantization...")
    
    try:
        # Load ground truth data
        ground_truth_path = env['data_dir'] / 'ground_truth.csv'
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Validate required columns
        required_columns = ['Invoice', 'Work Order Number/Numero de Orden', 'Total']
        missing_columns = [col for col in required_columns if col not in ground_truth_df.columns]
        if missing_columns:
            raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
        
        # Load model once for all test cases
        logger.info(f"Loading model {MODEL_NAME} with quantization {quant_level}")
        model = load_model(
            model_name=MODEL_NAME,
            quantization=quant_level,
            models_dir=env['models_dir'],
            config=config
        )
        
        # Group test cases by prompt type and field type
        grouped_cases = {}
        for case in quant_test_cases:
            key = (case['prompt_type'], case['field_type'])
            if key not in grouped_cases:
                grouped_cases[key] = []
            grouped_cases[key].append(case)
        
        # Run test suite for this quantization level
        results = []
        for (prompt_type, field_type), cases in grouped_cases.items():
            print(f"\nProcessing {len(cases)} cases with prompt type: {prompt_type}, field type: {field_type}")
            
            # Load prompt template once for this group
            prompt_template = load_prompt_template(
                prompt_strategy=prompt_type,
                prompts_dir=ROOT_DIR / "config" / "prompts"
            )
            
            # Process images one at a time
            for i, case in enumerate(cases):
                print(f"\nProcessing case {i + 1}/{len(cases)}")
                
                # Track execution start
                track_execution(
                    EXECUTION_LOG_PATH,
                    MODEL_NAME,
                    case['prompt_type'],
                    quant_level,
                    "started"
                )
                
                try:
                    # Get image ID from path
                    image_id = Path(case['image_path']).stem
                    
                    # Get ground truth for this image
                    image_ground_truth = ground_truth_df[ground_truth_df['Invoice'] == int(image_id)].iloc[0]
                    ground_truth = {
                        'work_order_number': image_ground_truth['Work Order Number/Numero de Orden'],
                        'total_cost': image_ground_truth['Total']
                    }
                    
                    # Process single image
                    result = process_image_wrapper(
                        model=model,
                        prompt_template=prompt_template,
                        image_path=str(case['image_path']),
                        field_type=field_type,
                        config=data_config
                    )
                    
                    # Evaluate results
                    evaluation = evaluate_model_output(
                        result['model_response']['output'],
                        ground_truth,
                        field_type
                    )
                    
                    # Add evaluation to result
                    result['evaluation'] = evaluation
                    
                    # Validate results
                    validated_result = validate_results(result)
                    
                    # Add test parameters and ground truth to result
                    validated_result['test_parameters'] = {
                        'model': MODEL_NAME,
                        'quantization': quant_level,
                        'prompt_strategy': prompt_template,
                        'image_path': str(case['image_path'])
                    }
                    validated_result['ground_truth'] = ground_truth
                    
                    results.append(validated_result)
                    print(f"✓ Completed test case {i + 1}/{len(cases)}")
                    
                    # Track successful completion
                    track_execution(
                        EXECUTION_LOG_PATH,
                        MODEL_NAME,
                        case['prompt_type'],
                        quant_level,
                        "completed"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing case: {str(e)}")
                    # Track failure
                    track_execution(
                        EXECUTION_LOG_PATH,
                        MODEL_NAME,
                        case['prompt_type'],
                        quant_level,
                        "failed",
                        str(e)
                    )
                    raise
        
        # Log all results for this quantization level in a single file
        if results:
            result_path = env['logs_dir'] / f"{MODEL_NAME}_{quant_level}bit_results.json"
            log_quantization_results(
                result_path=result_path,
                results=results,
                model_name=MODEL_NAME,
                prompt_type="all",  # Use "all" since we're combining all prompt types
                quant_level=quant_level
            )
            print(f"\n✓ All results logged to: {result_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in quantization level processing: {str(e)}")
        raise

# %% [markdown]
# ### Run 32-bit Quantization Tests
# 
# Run the test suite with full precision (32-bit) quantization.

# %%
# Load test matrix
with open(TEST_MATRIX_PATH, 'r') as f:
    test_matrix = json.load(f)

# Run 32-bit tests
results_32bit = run_quantization_level(32, test_matrix)

# %% [markdown]
# ### Run 16-bit Quantization Tests
# 
# Run the test suite with 16-bit quantization.

# %%
# Run 16-bit tests
results_16bit = run_quantization_level(16, test_matrix)

# %% [markdown]
# ### Run 8-bit Quantization Tests
# 
# Run the test suite with 8-bit quantization.

# %%
# Run 8-bit tests
results_8bit = run_quantization_level(8, test_matrix)

# %% [markdown]
# ### Run 4-bit Quantization Tests
# 
# Run the test suite with 4-bit quantization.

# %%
# Run 4-bit tests
results_4bit = run_quantization_level(4, test_matrix)

# %% [markdown]
# ## Results Analysis
# 
# Analyze and visualize the results for Pixtral model across all quantization levels.

# %%
def analyze_all_results():
    """Analyze results from all quantization levels."""
    all_results = []
    for quant_level, results in [
        (32, results_32bit),
        (16, results_16bit),
        (8, results_8bit),
        (4, results_4bit)
    ]:
        if results:
            all_results.extend(results)
    
    return analyze_results(all_results)

# Analyze all results
try:
    metrics = analyze_all_results()
    if metrics:
        print("\nPerformance Metrics by Quantization Level:")
        for quant_level, metric in metrics.items():
            print(f"\n{quant_level}-bit Quantization:")
            print(f"Work Order Accuracy: {metric['work_order_accuracy']:.2%}")
            print(f"Total Cost Accuracy: {metric['total_cost_accuracy']:.2%}")
            print(f"Average Processing Time: {metric['avg_processing_time']:.2f}s")
    else:
        print("No valid results to analyze")
except Exception as e:
    logger.error(f"Error analyzing results: {str(e)}")
    raise 