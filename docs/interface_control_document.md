# Interface Control Document: LMM Invoice Data Extraction Comparison

This document defines the key interfaces between components in the system, focusing on function signatures, data formats, and component interactions for rapid implementation.

## 1. Core Function Interfaces

### 1.1 Environment Setup

```python
def setup_environment(cuda_visible_devices=None, seed=42):
    """
    Set up environment for model execution.
    
    Args:
        cuda_visible_devices: str, GPU devices to use (e.g., "0,1")
        seed: int, random seed for reproducibility
        
    Returns:
        dict: Environment configuration
    """
    
def get_model_path(model_name, quant_level):
    """
    Get path to model with specific quantization.
    
    Args:
        model_name: str, name of the model
        quant_level: int, quantization level (4, 8, 16, 32)
        
    Returns:
        Path: Path to the model files
    """
```

### 1.2 Data Management

```python
def load_image(image_id, image_dir):
    """
    Load image by ID from directory.
    
    Args:
        image_id: str, image identifier
        image_dir: Path, directory containing images
        
    Returns:
        PIL.Image: Loaded image
    """
    
def load_ground_truth(csv_path):
    """
    Load ground truth data from CSV with both raw and normalized values.
    
    Args:
        csv_path: Path, path to ground truth CSV
        
    Returns:
        dict: Dictionary mapping image IDs to ground truth data with both raw and normalized values
              {
                image_id: {
                  "work_order_number": {
                    "raw_value": "20502",
                    "normalized_value": "20502"
                  },
                  "total_cost": {
                    "raw_value": " 950.00 ",
                    "normalized_value": 950.00
                  }
                },
                ...
              }
    """
    
def get_field_ground_truth(ground_truth, image_id, field_type):
    """
    Get ground truth for a specific field type and image.
    
    Args:
        ground_truth: dict, complete ground truth dictionary
        image_id: str, image identifier
        field_type: str, type of field to extract ("work_order_number" or "total_cost")
        
    Returns:
        dict: Ground truth for specific field
             {
               "raw_value": "20502",
               "normalized_value": "20502"
             }
    """
    
def normalize_cost(value):
    """
    Convert cost string to float, handling currency formatting.
    
    Args:
        value: str, cost value with potential formatting
        
    Returns:
        float: Normalized cost value
    """
```

### 1.3 Field-Specific Evaluation

```python
def evaluate_field_extraction(field_type, parsed_value, ground_truth):
    """
    Evaluate field extraction against ground truth.
    
    Args:
        field_type: str, type of field ("work_order_number" or "total_cost")
        parsed_value: str, extracted value from model
        ground_truth: dict, ground truth with raw and normalized values
                     {
                       "raw_value": "20502",
                       "normalized_value": "20502"
                     }
        
    Returns:
        dict: Evaluation metrics appropriate for field type
             {
               "raw_string_match": bool,
               "normalized_match": bool,
               "cer": float,
               "error_category": str
             }
    """
    
def calculate_cer(pred, true):
    """
    Calculate Character Error Rate between predicted and true values.
    
    Args:
        pred: str, predicted value
        true: str, true value
        
    Returns:
        float: Character Error Rate (0.0 to 1.0)
    """
    
def categorize_error(pred, true, field_type):
    """
    Categorize error type based on field and difference.
    
    Args:
        pred: str, predicted value
        true: str, true value
        field_type: str, type of field ("work_order_number" or "total_cost")
        
    Returns:
        str: Error category (e.g., "missing_character", "currency_error")
    """
```

### 1.4 Result Management

```python
def create_result_dict(test_params):
    """
    Create a new result dictionary for test parameters.
    
    Args:
        test_params: dict, parameters for the test
                    {
                      "model_name": "pixtral",
                      "field_type": "total_cost",
                      "prompt_type": "simple_total",
                      "quant_level": 4
                    }
        
    Returns:
        dict: Result dictionary with metadata and empty results section
    """
    
def add_image_result(result_dict, image_id, ground_truth, 
                     raw_text, parsed_value, normalized_value, 
                     processing_time, evaluation):
    """
    Add results for a specific image to the result dictionary.
    
    Args:
        result_dict: dict, result dictionary to update
        image_id: str, image identifier
        ground_truth: dict, ground truth for this field and image
        raw_text: str, raw model output text
        parsed_value: str, parsed field value from model output
        normalized_value: str or float, normalized field value
        processing_time: float, inference time in seconds
        evaluation: dict, evaluation metrics
        
    Returns:
        dict: Updated result dictionary
    """
    
def save_result(result_dict, results_dir):
    """
    Save result dictionary to JSON file.
    
    Args:
        result_dict: dict, complete result dictionary
        results_dir: Path, directory to save results
        
    Returns:
        str: Path to saved file
    """
    
def load_results(results_dir):
    """
    Load all result dictionaries from a directory.
    
    Args:
        results_dir: Path, directory containing result JSON files
        
    Returns:
        list: List of result dictionaries
    """
```

### 1.5 Model Interface

```python
def load_model(model_name, quant_level):
    """
    Load model with specific quantization.
    
    Args:
        model_name: str, name of the model
        quant_level: int, quantization level (4, 8, 16, 32)
        
    Returns:
        object: Loaded model
    """
    
def run_field_inference(model, image, prompt, field_type):
    """
    Run inference for a specific field type.
    
    Args:
        model: object, loaded model
        image: PIL.Image, input image
        prompt: str, formatted prompt for specific field type
        field_type: str, type of field to extract
        
    Returns:
        tuple: (raw_text, parsed_value, normalized_value, processing_time)
               raw_text: str, complete model output
               parsed_value: str, extracted field value
               normalized_value: str or float, field value in normalized form
               processing_time: float, inference time in seconds
    """
    
def parse_field_value(raw_text, field_type):
    """
    Extract field value from model output for specific field type.
    
    Args:
        raw_text: str, raw model output
        field_type: str, type of field to extract
        
    Returns:
        tuple: (parsed_value, normalized_value)
               parsed_value: str, extracted field value
               normalized_value: str or float, normalized field value
    """
```

### 1.6 Prompt Management

```python
def get_field_prompt_template(prompt_type, field_type):
    """
    Get prompt template for specific field type.
    
    Args:
        prompt_type: str, type of prompt (e.g., "simple", "detailed")
        field_type: str, type of field ("work_order_number" or "total_cost")
        
    Returns:
        str: Field-specific prompt template
    """
    
def format_field_prompt(template, model_name, field_type, **kwargs):
    """
    Format field-specific prompt template.
    
    Args:
        template: str, prompt template
        model_name: str, name of the model
        field_type: str, type of field to extract
        **kwargs: Additional keyword arguments for formatting
        
    Returns:
        str: Formatted field-specific prompt
    """
```

## 2. Result Dictionary Structure

```python
{
    "meta": {
        "experiment_id": "exp-20250410-123045",
        "timestamp": "2025-04-10T12:30:45",
        "environment": "RunPod T4 GPU"
    },
    "test_parameters": {
        "model_name": "pixtral",
        "field_type": "total_cost",
        "prompt_type": "simple_total",
        "quant_level": 4
    },
    "results_by_image": {
        "1017": {
            "ground_truth": {
                "raw_value": " 950.00 ",
                "normalized_value": 950.00
            },
            "model_response": {
                "raw_text": "Total Cost: $950.00",
                "parsed_value": "$950.00",
                "normalized_value": 950.00,
                "processing_time": 1.23
            },
            "evaluation": {
                "raw_string_match": false,
                "normalized_match": true,
                "cer": 0.25,
                "error_category": "currency_format"
            }
        },
        "1018": {
            "ground_truth": {
                "raw_value": " 550.00 ",
                "normalized_value": 550.00
            },
            "model_response": {
                "raw_text": "Total Cost: $550.00",
                "parsed_value": "$550.00",
                "normalized_value": 550.00,
                "processing_time": 1.19
            },
            "evaluation": {
                "raw_string_match": false,
                "normalized_match": true,
                "cer": 0.25,
                "error_category": "currency_format"
            }
        },
        // More images...
    }
}
```

## 3. File Formats

### 3.1 Ground Truth CSV Format

```
Invoice,Type,Timestamp,Name,Work Order Number/Numero de Orden,Total
1017,Invoice,10/17/2024,Edgar,20502, 950.00 
1018,Invoice,10/17/2024,Edgar,20558, 550.00 
1019,Invoice,10/17/2024,Edgar,20509, 150.00 
...
```

### 3.2 Field-Specific Test Matrix CSV Format

```
model_name,field_type,prompt_type,quant_level
pixtral,work_order_number,simple_wo,4
pixtral,work_order_number,detailed_wo,4
pixtral,total_cost,simple_total,4
pixtral,total_cost,detailed_total,4
...
```

## 4. Analysis Functions

```python
def filter_results(results, **kwargs):
    """
    Filter results by any parameter.
    
    Args:
        results: list, result dictionaries
        **kwargs: Key-value pairs for filtering (e.g., model_name="pixtral")
        
    Returns:
        list: Filtered result dictionaries
    """
    
def aggregate_by_model_field(results):
    """
    Aggregate results by model and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by model and field type
             {
               "pixtral": {
                 "work_order_number": {"total": 100, "matches": 80, "avg_cer": 0.1},
                 "total_cost": {"total": 100, "matches": 75, "avg_cer": 0.15}
               },
               "llama": {
                 "work_order_number": {"total": 100, "matches": 85, "avg_cer": 0.05},
                 "total_cost": {"total": 100, "matches": 70, "avg_cer": 0.2}
               }
             }
    """
    
def aggregate_by_prompt_field(results):
    """
    Aggregate results by prompt type and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by prompt type and field type
    """
    
def aggregate_by_quant_field(results):
    """
    Aggregate results by quantization level and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by quantization level and field type
    """
    
def calculate_error_distribution(results, field_type):
    """
    Calculate distribution of error categories for specific field type.
    
    Args:
        results: list, result dictionaries
        field_type: str, field type to analyze
        
    Returns:
        dict: Error category distribution
             {
               "currency_error": 15,
               "digit_error": 10,
               "decimal_error": 5,
               ...
             }
    """
```

## 5. Notebook Flow

Each model-specific notebook follows this field-specific flow:

```python
# 1. Set up environment
env_config = setup_environment()

# 2. Load test configurations
test_matrix = load_test_matrix("config/test_matrix.csv")
model_tests = filter_test_matrix(test_matrix, model_name="pixtral")
all_ground_truth = load_ground_truth("data/ground_truth.csv")

# 3. For each test configuration
for test_params in model_tests:
    # Create result dictionary for this configuration
    result_dict = create_result_dict(test_params)
    
    # Load model with specified quantization
    model = load_model(test_params["model_name"], test_params["quant_level"])
    
    # Get field-specific prompt template
    field_type = test_params["field_type"]
    prompt_type = test_params["prompt_type"]
    template = get_field_prompt_template(prompt_type, field_type)
    
    # For each test image
    for image_id in get_test_image_ids():
        # Get ground truth for this field and image
        gt = get_field_ground_truth(all_ground_truth, image_id, field_type)
        
        # Load image
        image = load_image(image_id, "data/images")
        
        # Format field-specific prompt
        prompt = format_field_prompt(template, test_params["model_name"], field_type)
        
        # Run inference for specific field
        raw_text, parsed_value, normalized_value, processing_time = (
            run_field_inference(model, image, prompt, field_type)
        )
        
        # Evaluate extraction for this field
        evaluation = evaluate_field_extraction(field_type, parsed_value, gt)
        
        # Add result for this image
        add_image_result(
            result_dict, image_id, gt, 
            raw_text, parsed_value, normalized_value, 
            processing_time, evaluation
        )
    
    # Save results for this configuration
    save_result(result_dict, "results/")
    
    # Unload model to free memory
    del model
```

The analysis notebook follows this field-specific flow:

```python
# 1. Load all results
results = load_results("results/")

# 2. Calculate field-specific metrics
model_field_metrics = aggregate_by_model_field(results)
prompt_field_metrics = aggregate_by_prompt_field(results)
quant_field_metrics = aggregate_by_quant_field(results)

# 3. Analyze error distributions by field type
wo_errors = calculate_error_distribution(results, "work_order_number")
cost_errors = calculate_error_distribution(results, "total_cost")

# 4. Generate field-specific visualizations
plot_model_field_comparison(model_field_metrics)
plot_prompt_field_comparison(prompt_field_metrics)
plot_quant_field_comparison(quant_field_metrics)
plot_error_distributions(wo_errors, cost_errors)

# 5. Identify best configurations for each field type
best_wo_config = find_best_configuration(results, "work_order_number")
best_cost_config = find_best_configuration(results, "total_cost")
```

## 6. Component Integration Flow

1. Config Layer provides:
   - Field-specific test matrix configurations
   - Field-specific prompt templates

2. Data Layer provides:
   - Image loading
   - Ground truth data with raw and normalized values
   - Field-specific ground truth access

3. Model Layer handles:
   - Model loading with quantization
   - Field-specific inference execution
   - Field-specific output parsing

4. Result Layer manages:
   - Creating result dictionaries organized by test parameters
   - Storing results by image
   - Field-appropriate evaluation metrics
   - Saving complete results

5. Analysis Layer provides:
   - Field-specific result aggregation
   - Field-specific metric calculation
   - Field-specific visualization generation
   - Best configuration identification for each field type

This functional approach with field-specific processing and dictionary data structures supports efficient testing and analysis while maintaining simplicity for rapid implementation.

## Configuration Management Interface

### src/config.py

#### load_yaml_config
```python
def load_yaml_config(config_path: str) -> Dict[str, Any]
```
Loads and parses YAML configuration file.

**Parameters:**
- `config_path`: Path to YAML configuration file

**Returns:**
- Dictionary containing configuration parameters

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If YAML parsing fails

#### parse_test_matrix
```python
def parse_test_matrix(csv_path: str) -> List[Dict[str, Any]]
```
Parses test matrix CSV into list of test cases.

**Parameters:**
- `csv_path`: Path to test matrix CSV file

**Returns:**
- List of dictionaries containing test case parameters

**Raises:**
- `FileNotFoundError`: If CSV file doesn't exist
- `pd.errors.EmptyDataError`: If CSV is empty

#### setup_logging_config
```python
def setup_logging_config(config: Dict[str, Any], log_dir: Path) -> Dict[str, Any]
```
Sets up logging configuration from main config.

**Parameters:**
- `config`: Main configuration dictionary
- `log_dir`: Path to logging directory (provided by environment setup)

**Returns:**
- Dictionary containing logging configuration

**Raises:**
- `KeyError`: If required logging config keys are missing

#### validate_config
```python
def validate_config(config: Dict[str, Any]) -> bool
```
Validates configuration structure and required fields.

**Parameters:**
- `config`: Configuration dictionary to validate

**Returns:**
- True if configuration is valid

**Raises:**
- `ValueError`: If configuration is invalid