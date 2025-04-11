# Project Directory Structure

## Overview
This document describes the project's directory structure and the purpose of each component.

## Root Directory
```
invoice-extraction-comparison/
├── config/                         # Configuration files
│   ├── models/                     # Model-specific configurations
│   │   ├── doctr.yaml             # Doctr model configuration
│   │   ├── llama_vision.yaml      # Llama Vision model configuration
│   │   └── pixtral.yaml           # Pixtral model configuration
│   ├── prompts/                    # Field-specific prompt templates
│   │   ├── basic_extraction.yaml  # Basic extraction prompt
│   │   ├── detailed.yaml          # Detailed extraction prompt
│   │   ├── few_shot.yaml          # Few-shot example prompt
│   │   ├── step_by_step.yaml      # Step-by-step prompt
│   │   └── locational.yaml        # Locational prompt
│   └── test_matrix.csv            # Test configuration matrix
├── data/                           # Data storage (gitignored)
├── src/                            # Source code modules
│   ├── environment.py              # Environment setup and paths
│   ├── config.py                   # Configuration management
│   ├── execution.py                # Test execution framework
│   └── models/                     # Model implementations
├── notebooks/                      # Jupyter notebooks
│   └── model_notebook_template.py  # Template for model evaluation
├── results/                        # Results storage (gitignored)
└── docs/                           # Documentation
    ├── adr/                        # Architecture Decision Records
    │   ├── 001-configuration-management.md
    │   ├── 002-prompt-strategy.md
    │   └── 003-test-matrix-execution.md
    ├── project_overview.md         # Project goals and scope
    ├── project_rules.md            # Implementation guidelines
    ├── project_todo.md             # Task list and progress tracking
    ├── interface_control_document.md # Component interfaces
    └── project_dir.md              # Detailed directory structure
```

## Directory Descriptions

### config/
Contains all configuration files for the project:
- `models/`: Model-specific configurations
- `prompts/`: Prompt templates for different strategies
- `test_matrix.csv`: Test configuration matrix

### src/
Core source code modules:
- `environment.py`: Environment setup
- `config.py`: Configuration management
- `execution.py`: Test execution framework
- `models/`: Model implementations

### notebooks/
Jupyter notebooks for evaluation:
- `model_notebook_template.py`: Template for model evaluation
- Future model-specific notebooks will be added here

## Configuration Management
The project uses a dependency injection pattern for configuration management:

1. **Environment Setup** (`src/environment.py`)
   - Sets up project paths
   - Configures CUDA
   - Installs dependencies
   - Creates necessary directories

2. **Configuration Management** (`src/config.py`)
   - YAML configuration loading
   - Test matrix CSV parsing
   - Logging configuration setup
   - Configuration validation

3. **Configuration Files** (`config/`)
   - Model-specific configurations
   - Prompt templates
   - Test matrix definitions

## Source Code
The `src/` directory contains the core implementation:

1. **Environment Module** (`environment.py`)
   - Path management
   - CUDA setup
   - Dependency installation

2. **Configuration Module** (`config.py`)
   - YAML parsing
   - CSV parsing
   - Configuration validation
   - Logging setup

3. **Model Implementations** (`models/`)
   - Model-specific code
   - Inference functions
   - Output parsing

## Documentation
The `docs/` directory contains project documentation:

1. **Architecture Decisions** (`adr/`)
   - ADR 001: Configuration Management Design

2. **Project Documentation**
   - Project overview
   - Implementation rules
   - Task tracking
   - Interface definitions
   - Directory structure

## Data and Results
The `data/` and `results/` directories are gitignored:

1. **Data Directory**
   - Input images
   - Ground truth CSV
   - Test matrices

2. **Results Directory**
   - Evaluation results
   - Log files
   - Analysis outputs

## Notebooks
The `notebooks/` directory contains execution environments:

1. **Model Evaluation**
   - Pixtral evaluation
   - Llama Vision evaluation
   - Doctr evaluation

2. **Results Analysis**
   - Comparative analysis
   - Visualization
   - Best model identification

## Key File Contents and Formats

### Configuration Files

#### Test Matrix CSV
```csv
model_name,field_type,prompt_type,quant_level
pixtral,work_order_number,simple,4
pixtral,work_order_number,detailed,4
pixtral,total_cost,simple,4
pixtral,total_cost,detailed,4
...
```

#### Field-Specific Prompt Template
```yaml
# config/prompts/work_order/simple.yaml
name: "simple_work_order"
description: "Simple work order extraction prompt"
template: |
  Look at this invoice image and extract the Work Order Number.
  The Work Order Number is typically in the format of a 5-digit numeric or alphanumeric code.
  Return only the exact Work Order Number, preserving all digits, letters, and formatting.
```

```yaml
# config/prompts/total_cost/simple.yaml
name: "simple_total_cost"
description: "Simple total cost extraction prompt"
template: |
  Look at this invoice image and extract the Total Cost.
  The Total Cost is typically shown as a dollar amount, often preceded by a $ symbol.
  Return only the exact Total Cost value as shown on the invoice.
```

### Source Code Files

#### data_utils.py
Contains functions for loading ground truth with both raw and normalized values:
```python
def load_ground_truth(csv_path):
    """Load ground truth data with both raw and normalized values."""
    # Implementation that creates dictionary with raw and normalized values
    # for each field type and image
```

#### evaluation.py
Contains field-specific evaluation functions:
```python
def evaluate_field_extraction(field_type, parsed_value, ground_truth):
    """Evaluate extraction for a specific field type."""
    # Implementation that applies appropriate evaluation metrics
    # based on field type (work_order_number or total_cost)
```

#### logging.py
Contains result dictionary management functions:
```python
def create_result_dict(test_params):
    """Create result dictionary grouped by test parameters."""
    # Implementation that creates dictionary structure

def add_image_result(result_dict, image_id, ground_truth, 
                     raw_text, parsed_value, normalized_value, 
                     processing_time, evaluation):
    """Add results for a specific image to the result dictionary."""
    # Implementation that adds image result to the dictionary
```

### Results Files

#### Field-Specific Result Dictionary
```json
{
    "meta": {
        "experiment_id": "exp-20250410-123045",
        "timestamp": "2025-04-10T12:30:45",
        "environment": "RunPod T4 GPU"
    },
    "test_parameters": {
        "model_name": "pixtral",
        "field_type": "total_cost",
        "prompt_type": "simple",
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
            /* More image results */
        }
    }
}
```

## Key Directory Relationships

### Configuration Management
- Field-specific prompt templates in `config/prompts/[field_type]/`
- Test matrix includes field type as a dimension
- Model configurations are field-agnostic

### Source Code Organization
- Core utilities are field-aware but not field-specific
- Model implementations include field-specific extraction logic
- Evaluation includes field-specific metrics

### Result Organization
- Results are organized first by field type
- Each result file contains results for one combination of (model, field, prompt, quant)
- Within each file, results are organized by image ID

## Implementation Notes

1. **Field Isolation**: Each field type is tested separately with field-specific prompts
2. **Dual Storage**: Both raw and normalized values are stored for comprehensive analysis
3. **Efficient Storage**: Results are grouped by test parameters to reduce duplication
4. **Field-Specific Analysis**: Analysis is performed separately for each field type
5. **Clear Organization**: Directory structure reflects the field-specific approach

This directory structure supports the field-specific testing approach while maintaining a clean organization that facilitates rapid implementation and analysis.

- **models/common.py**: Shared model utilities for preprocessing and postprocessing
- **models/[model_name].py**: Model-specific implementation files with load and inference functions
- **visualization.py**: Result visualization utilities for analysis notebook

### Notebooks

Each model notebook follows the same structure:
1. **Import dependencies**
2. **Configure environment**
3. **Load model-specific test configurations**
4. **Execute tests for all prompt/quantization combinations**
5. **Log results**

The analysis notebook:
1. **Load and aggregate results**
2. **Generate comparative visualizations**
3. **Identify best-performing combinations**
4. **Create summary tables and charts**

### Results

- **results.csv**: Main results log with metrics for all tests
- **errors.csv**: Error log for debugging
- **charts/**: Generated visualization files

## Module Dependencies

```
environment.py <- config.py <- data_utils.py <- models/ <- evaluation.py <- logging.py
                                                  ^
                                                  |
                                               prompts.py
```

Model notebooks depend on all source modules:
```
model_notebook.ipynb <- environment.py, config.py, data_utils.py, 
                        models/, prompts.py, evaluation.py, logging.py
```

Analysis notebook depends on results and visualization:
```
results_analysis.ipynb <- logging.py, visualization.py
```

## Implementation Notes

1. **Flat structure**: Minimizes directory nesting for simpler navigation
2. **Functional organization**: Groups code by functionality rather than architecture
3. **Clear separation**: Keeps configuration, code, and data clearly separated
4. **Parallel execution**: Supports running model notebooks in parallel
5. **Simplified imports**: Uses direct imports rather than complex package structure

This directory structure is designed for rapid implementation while maintaining good organization and separation of concerns. Unlike the more complex structure in the original plan, this version focuses on a functional approach that can be implemented quickly while still producing reliable results.