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

This functional approach with field-specific processing and dictionary data structures supports efficient testing and analysis while maintaining simplicity for rapid implementation.# Interface Control Document: LMM Invoice Data Extraction Comparison

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
    Load ground truth data from CSV preserving original formats.
    
    Args:
        csv_path: Path, path to ground truth CSV
        
    Returns:
        dict: Dictionary mapping image IDs to ground truth data
              {image_id: {"work_order_number": "20502", "total_cost": " 950.00 "}}
    """
    
def normalize_field_names(prediction):
    """
    Normalize field names from model output without changing values.
    
    Args:
        prediction: dict, model output with various field names
        
    Returns:
        dict: Dictionary with standardized field names
             {"work_order_number": "20502", "total_cost": "$950.00"}
    """
```

### 1.3 Evaluation Functions

```python
def evaluate_extraction(prediction, ground_truth):
    """
    Compare prediction to ground truth with exact string matching.
    
    Args:
        prediction: dict, extracted fields from model
                   {"work_order_number": "20502", "total_cost": "$950.00"}
        ground_truth: dict, ground truth fields
                      {"work_order_number": "20502", "total_cost": " 950.00 "}
        
    Returns:
        dict: Evaluation metrics
              {
                "overall": {"exact_match": bool},
                "fields": {
                  "work_order_number": {
                    "exact_match": bool,
                    "cer": float,
                    "error_category": str
                  },
                  "total_cost": {
                    "exact_match": bool,
                    "cer": float,
                    "error_category": str
                  }
                }
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

### 1.4 Result Management Functions

```python
def create_result_dict(test_params, ground_truth):
    """
    Create a new result dictionary with metadata.
    
    Args:
        test_params: dict, parameters of the test
                    {"model_name": "pixtral", "prompt_type": "direct", "quant_level": 4}
        ground_truth: dict, ground truth data for the image
                     {"work_order_number": "20502", "total_cost": " 950.00 "}
        
    Returns:
        dict: Result dictionary with metadata and empty response fields
    """
    
def update_result_with_response(result_dict, raw_text, parsed_fields, processing_time):
    """
    Update result dictionary with model response.
    
    Args:
        result_dict: dict, result dictionary to update
        raw_text: str, raw text output from model
        parsed_fields: dict, extracted fields from raw text
        processing_time: float, inference time in seconds
        
    Returns:
        dict: Updated result dictionary
    """
    
def evaluate_and_update_result(result_dict):
    """
    Evaluate extraction and update result dictionary with evaluation metrics.
    
    Args:
        result_dict: dict, result dictionary with model response
        
    Returns:
        dict: Updated result dictionary with evaluation results
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
```

### 1.5 Model Interfaces

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
    
def run_inference(model, image, prompt):
    """
    Run inference with model and return raw output.
    
    Args:
        model: object, loaded model
        image: PIL.Image, input image
        prompt: str, formatted prompt
        
    Returns:
        tuple: (raw_text, processing_time)
               raw_text: str, raw model output
               processing_time: float, inference time in seconds
    """
    
def parse_model_output(raw_text, model_name):
    """
    Parse raw model output to extract structured fields.
    
    Args:
        raw_text: str, raw text from model
        model_name: str, name of the model
        
    Returns:
        dict: Extracted fields with original formatting
              {"Work Order Number": "20502", "Total Cost": "$950.00"}
    """
```

### 1.6 Prompt Management

```python
def get_prompt_template(prompt_type):
    """
    Get prompt template by type.
    
    Args:
        prompt_type: str, type of prompt
        
    Returns:
        str: Prompt template
    """
    
def format_prompt(template, model_name, **kwargs):
    """
    Format prompt template for specific model.
    
    Args:
        template: str, prompt template
        model_name: str, name of the model
        **kwargs: Additional keyword arguments for formatting
        
    Returns:
        str: Formatted prompt
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
        "prompt_type": "direct",
        "quant_level": 4,
        "image_id": "1017"
    },
    "ground_truth": {
        "work_order_number": "20502",
        "total_cost": " 950.00 "
    },
    "model_response": {
        "raw_text": "Work Order Number: 20502\nTotal Cost: $950.00",
        "parsed_fields": {
            "work_order_number": "20502",
            "total_cost": "$950.00"
        },
        "processing_time": 1.23
    },
    "evaluation_results": {
        "overall": {
            "exact_match": false
        },
        "fields": {
            "work_order_number": {
                "exact_match": true,
                "cer": 0.0,
                "error_category": null
            },
            "total_cost": {
                "exact_match": false,
                "cer": 0.25,
                "error_category": "currency_error"
            }
        }
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

### 3.2 Test Matrix CSV Format

```
model_name,prompt_type,quant_level
pixtral,direct,4
pixtral,direct,8
pixtral,detailed,4
...
```

## 4. Analysis Functions

```python
def load_results(results_dir):
    """
    Load all result dictionaries from a directory.
    
    Args:
        results_dir: Path, directory containing result JSON files
        
    Returns:
        list: List of result dictionaries
    """
    
def filter_results(results, **kwargs):
    """
    Filter results by any parameter.
    
    Args:
        results: list, result dictionaries
        **kwargs: Key-value pairs for filtering
                 (e.g., model_name="pixtral")
        
    Returns:
        list: Filtered result dictionaries
    """
    
def aggregate_by_model(results):
    """
    Aggregate results by model.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Aggregated metrics by model
              {
                "pixtral": {"total": 100, "exact_matches": 75},
                "llama": {"total": 100, "exact_matches": 80},
                ...
              }
    """
    
def calculate_field_metrics(results, field_type):
    """
    Calculate metrics for a specific field type.
    
    Args:
        results: list, result dictionaries
        field_type: str, field to analyze ("work_order_number" or "total_cost")
        
    Returns:
        dict: Field-specific metrics
              {
                "exact_match_rate": 0.75,
                "avg_cer": 0.15,
                "error_categories": {
                  "currency_error": 10,
                  "digit_error": 15,
                  ...
                }
              }
    """
```

## 5. Notebook Flow

Each model-specific notebook follows this standard flow:

```python
# 1. Set up environment
env_config = setup_environment()

# 2. Load test configurations
test_matrix = load_test_matrix("config/test_matrix.csv")
model_tests = filter_test_matrix(test_matrix, model_name="pixtral")
ground_truth = load_ground_truth("data/ground_truth.csv")

# 3. For each test configuration
for test_params in model_tests:
    # Load model with specified quantization
    model = load_model(test_params["model_name"], test_params["quant_level"])
    
    # Get prompt template
    template = get_prompt_template(test_params["prompt_type"])
    
    # For each test image
    for image_id in get_test_image_ids():
        # Create result dictionary
        result = create_result_dict(
            {**test_params, "image_id": image_id},
            ground_truth[image_id]
        )
        
        # Load image
        image = load_image(image_id, "data/images")
        
        # Format prompt
        prompt = format_prompt(template, test_params["model_name"])
        
        # Run inference
        raw_text, processing_time = run_inference(model, image, prompt)
        
        # Parse output
        parsed_fields = parse_model_output(raw_text, test_params["model_name"])
        
        # Update result with response
        result = update_result_with_response(result, raw_text, parsed_fields, processing_time)
        
        # Evaluate result
        result = evaluate_and_update_result(result)
        
        # Save result
        save_result(result, "results/")
    
    # Unload model to free memory
    del model
```

The analysis notebook follows this flow:

```python
# 1. Load all results
results = load_results("results/")

# 2. Filter results as needed
pixtral_results = filter_results(results, test_parameters__model_name="pixtral")
direct_prompt_results = filter_results(results, test_parameters__prompt_type="direct")

# 3. Calculate aggregated metrics
model_metrics = aggregate_by_model(results)
prompt_metrics = aggregate_by_prompt_type(results)
quant_metrics = aggregate_by_quant_level(results)

# 4. Calculate field-specific metrics
work_order_metrics = calculate_field_metrics(results, "work_order_number")
total_cost_metrics = calculate_field_metrics(results, "total_cost")

# 5. Generate visualizations
plot_model_comparison(model_metrics)
plot_prompt_comparison(prompt_metrics)
plot_quant_comparison(quant_metrics)
plot_error_categories(work_order_metrics, total_cost_metrics)
```

## 6. Component Integration Flow

1. Config Layer provides:
   - Test matrix configurations
   - Model parameters
   - Prompt templates

2. Data Layer provides:
   - Image loading
   - Ground truth data access
   - Field name normalization

3. Model Layer handles:
   - Model loading with quantization
   - Inference execution
   - Output parsing

4. Result Layer manages:
   - Creating result dictionaries
   - Evaluating extraction accuracy
   - Calculating error metrics
   - Saving complete results

5. Analysis Layer provides:
   - Result aggregation
   - Metric calculation
   - Visualization generation
   - Best model identification

This functional approach using dictionary data structures for results provides comprehensive context for analysis while maintaining simplicity for rapid implementation.# Interface Control Document: LMM Invoice Data Extraction Comparison

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
def load_ground_truth(csv_path):
    """
    Load ground truth data from CSV.
    
    Args:
        csv_path: Path, path to ground truth CSV
        
    Returns:
        dict: Dictionary mapping image IDs to ground truth data
              {image_id: {"Work Order Number": "12345", "Total Cost": 123.45}}
              
    Notes:
        - Handles type conversions from original CSV format
        - Normalizes column names for consistent access
        - Properly converts 'Total' from string with currency formatting to float
        - Ensures 'Work Order Number' is stored as string to preserve formatting (e.g., leading zeros)
    """
    
def normalize_work_order(text):
    """
    Normalize work order text for comparison.
    
    Args:
        text: str, work order text to normalize
        
    Returns:
        str: Normalized work order text
        
    Notes:
        - Handles alphanumeric work order numbers (e.g., "20502", "Aston")
        - Trims whitespace
        - Preserves original case for comparison
        - Handles special cases (e.g., null or missing values)
    """
    
def normalize_cost(text):
    """
    Normalize cost values for comparison.
    
    Args:
        text: str or float, cost value to normalize
        
    Returns:
        float: Normalized cost value
        
    Notes:
        - Handles various cost formats ("$1,234.56", "1,234.56", "1234.56", 1234.56)
        - Removes currency symbols ($)
        - Removes thousands separators (,)
        - Handles trailing/leading whitespace
        - Converts to float for numeric comparison
    """
    
def evaluate_extraction(prediction, ground_truth):
    """
    Compare prediction to ground truth.
    
    Args:
        prediction: dict, extracted fields from model
                   {"Work Order Number": "12345", "Total Cost": "1,234.56"}
        ground_truth: dict, ground truth fields
                      {"Work Order Number": "12345", "Total Cost": "1,234.56"}
        
    Returns:
        dict: Evaluation metrics
              {
                "exact_match": bool,
                "work_order_match": bool, 
                "total_cost_match": bool,
                "work_order_cer": float,  # Character Error Rate if not exact match
                "total_cost_cer": float,  # Character Error Rate if not exact match
                "error_category": str     # Categorized error type
              }
              
    Notes:
        - Uses exact string comparison (no normalization)
        - Character Error Rate calculated for non-matches
        - Error categories include:
          - For Work Order: "missing_digit", "wrong_character", "transposition", "extra_character"
          - For Total Cost: "currency_error", "decimal_error", "digit_error", "formatting_error"
        - Preserves original string formats for comparison
    """
```

### 5.5 Results CSV Format

```
model_name,prompt_type,quant_level,image_id,exact_match,work_order_match,total_cost_match,work_order_cer,total_cost_cer,error_category,timestamp
pixtral,direct,4,12147,True,True,True,0.0,0.0,,2023-03-01T12:30:45
pixtral,direct,4,12148,False,True,False,0.0,0.25,digit_error,2023-03-01T12:31:10
...
```

### 1.3 Logging Functions

```python
def log_result(model_name, prompt_type, quant_level, image_id, metrics, log_file):
    """
    Append result to CSV log file.
    
    Args:
        model_name: str, name of the model
        prompt_type: str, type of prompt used
        quant_level: int, quantization level
        image_id: str, image identifier
        metrics: dict, evaluation metrics
        log_file: Path, path to log file
    """
    
def already_tested(model_name, prompt_type, quant_level, image_id, log_file):
    """
    Check if a specific combination was already tested.
    
    Args:
        model_name: str, name of the model
        prompt_type: str, type of prompt used
        quant_level: int, quantization level
        image_id: str, image identifier
        log_file: Path, path to log file
        
    Returns:
        bool: True if already tested, False otherwise
    """
```

## 2. Model Interfaces

### 2.1 Model Registry Interface

```python
# models/__init__.py
def get_model(model_name):
    """
    Get model class by name.
    
    Args:
        model_name: str, name of the model
        
    Returns:
        Module: Model module containing load and run_inference functions
    """
```

### 2.2 Model-Specific Interfaces

Each model module must implement these functions:

```python
def load_model(quant_level):
    """
    Load model with specific quantization.
    
    Args:
        quant_level: int, quantization level (4, 8, 16, 32)
        
    Returns:
        Model: Loaded model object
    """
    
def run_inference(model, image, prompt):
    """
    Run inference with specific model.
    
    Args:
        model: Model, loaded model object
        image: PIL.Image, image to process
        prompt: str, formatted prompt
        
    Returns:
        dict: Extracted fields
              {"Work Order Number": "12345", "Total Cost": 123.45}
    """
```

## 3. Prompt Management Interface

```python
def get_prompt_template(prompt_type):
    """
    Get prompt template by type.
    
    Args:
        prompt_type: str, type of prompt
        
    Returns:
        str: Prompt template
    """
    
def format_prompt(template, model_name, **kwargs):
    """
    Format prompt template for specific model.
    
    Args:
        template: str, prompt template
        model_name: str, name of the model
        **kwargs: Additional keyword arguments for formatting
        
    Returns:
        str: Formatted prompt
    """
```

## 4. Configuration Interface

```python
def load_test_matrix(csv_path):
    """
    Load test matrix from CSV.
    
    Args:
        csv_path: Path, path to test matrix CSV
        
    Returns:
        list: List of test configurations
              [{"model_name": "pixtral", "prompt_type": "direct", "quant_level": 4}, ...]
    """
    
def load_model_config(model_name, config_dir):
    """
    Load model configuration from YAML.
    
    Args:
        model_name: str, name of the model
        config_dir: Path, directory containing configurations
        
    Returns:
        dict: Model configuration
    """
    
def load_prompt_config(prompt_type, config_dir):
    """
    Load prompt configuration from YAML.
    
    Args:
        prompt_type: str, type of prompt
        config_dir: Path, directory containing configurations
        
    Returns:
        dict: Prompt configuration
    """
```

## 5. Data Formats

### 5.1 Ground Truth CSV Format

```
image_id,Work Order Number,Total Cost
12147,AB123,1234.56
12148,CD456,789.01
...
```

### 5.2 Test Matrix CSV Format

```
model_name,prompt_type,quant_level
pixtral,direct,4
pixtral,direct,8
pixtral,detailed,4
...
```

### 5.3 Results CSV Format

```
model_name,prompt_type,quant_level,image_id,exact_match,work_order_match,total_cost_match,timestamp
pixtral,direct,4,12147,True,True,True,2023-03-01T12:30:45
pixtral,direct,4,12148,False,True,False,2023-03-01T12:31:10
...
```

### 5.4 Model Output Format

All model outputs should be normalized to this format:

```python
{
    "Work Order Number": "AB123",  # String, preserves format
    "Total Cost": 1234.56         # Float, normalized
}
```

## 6. Notebook Interface

Each model-specific notebook follows this interface pattern:

1. **Import dependencies** from `src/` modules
2. **Load configuration** for model and prompts
3. **Get test matrix** for specific model
4. For each test configuration:
   - **Load model** with specified quantization
   - For each test image:
     - **Check if already tested**
     - **Load image**
     - **Format prompt**
     - **Run inference**
     - **Evaluate results**
     - **Log results**
   - **Unload model** to free memory

The results analysis notebook follows this interface pattern:

1. **Load all results** from results CSV
2. **Aggregate results** by model/prompt/quantization
3. **Calculate metrics** for each combination
4. **Visualize comparisons** across combinations
5. **Identify best combination** based on metrics

## 7. Error Handling Interface

```python
def log_error(model_name, prompt_type, quant_level, image_id, error, log_file):
    """
    Log error to error log file.
    
    Args:
        model_name: str, name of the model
        prompt_type: str, type of prompt used
        quant_level: int, quantization level
        image_id: str, image identifier
        error: Exception, error that occurred
        log_file: Path, path to error log file
    """
```

## 8. Component Integration Flow

1. Environment setup → Model loading
2. Configuration loading → Test matrix generation
3. Image loading + Prompt formatting → Model inference
4. Model output + Ground truth → Evaluation
5. Evaluation metrics → Results logging
6. Aggregated results → Analysis and visualization

This interface document provides the contract between components, ensuring consistent integration despite the rapid implementation focus.