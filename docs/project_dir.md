# Project Directory Structure: LMM Invoice Data Extraction Comparison (Rapid Implementation)

This document outlines the directory structure for the field-specific version of the project, optimized for the 8-hour development timeline.

```
invoice-extraction-comparison/
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Python dependencies
│
├── config/                         # Configuration files
│   ├── models/                     # Model-specific configurations
│   │   ├── pixtral.yaml            # Pixtral-12B configuration
│   │   ├── llama_vision.yaml       # Llama-3.2-11B-Vision configuration
│   │   └── doctr.yaml              # Doctr configuration
│   ├── prompts/                    # Field-specific prompt templates
│   │   ├── work_order/             # Work order number prompts
│   │   │   ├── simple.yaml         # Simple work order prompt
│   │   │   ├── detailed.yaml       # Detailed work order prompt
│   │   │   └── location.yaml       # Locational work order prompt
│   │   ├── total_cost/             # Total cost prompts
│   │   │   ├── simple.yaml         # Simple total cost prompt
│   │   │   ├── detailed.yaml       # Detailed total cost prompt
│   │   │   └── location.yaml       # Locational total cost prompt
│   └── test_matrix.csv             # Field-specific test combination matrix
│
├── data/                           # Data storage (gitignored)
│   ├── images/                     # Invoice images
│   ├── ground_truth.csv            # Ground truth data
│   └── README.md                   # Data directory documentation
│
├── src/                            # Source code modules
│   ├── environment.py              # Environment setup utilities
│   ├── config.py                   # Configuration loading utilities
│   ├── data_utils.py               # Data loading and processing with raw/normalized values
│   ├── evaluation.py               # Field-specific evaluation metrics
│   ├── logging.py                  # Result dictionary management
│   ├── prompts.py                  # Field-specific prompt management
│   ├── models/                     # Model implementations
│   │   ├── __init__.py             # Model registry and factory
│   │   ├── common.py               # Shared model utilities
│   │   ├── pixtral.py              # Pixtral model with field-specific extraction
│   │   ├── llama_vision.py         # Llama Vision model with field-specific extraction
│   │   └── doctr.py                # Doctr model with field-specific extraction
│   └── visualization.py            # Field-specific result visualization
│
├── notebooks/                      # Jupyter notebooks
│   ├── pixtral_evaluation.ipynb    # Pixtral model evaluation
│   ├── llama_vision_evaluation.ipynb # Llama Vision model evaluation
│   ├── doctr_evaluation.ipynb      # Doctr model evaluation
│   └── results_analysis.ipynb      # Field-specific result analysis
│
├── results/                        # Results storage (gitignored)
│   ├── work_order/                 # Work order field results
│   │   ├── pixtral_simple_4.json   # Example result file for work order field
│   │   └── ...                     # Other result files
│   ├── total_cost/                 # Total cost field results
│   │   ├── pixtral_simple_4.json   # Example result file for total cost field
│   │   └── ...                     # Other result files
│   └── charts/                     # Generated visualizations
│       ├── work_order_comparison.png # Field-specific comparison chart
│       ├── total_cost_comparison.png # Field-specific comparison chart
│       └── ...                     # Other visualization files
│
└── docs/                           # Documentation
    ├── project-overview.md         # Project overview
    ├── project-rules.md            # Implementation rules
    ├── project-todo.md             # Task list
    ├── interface_control_document.md # Component interfaces
    ├── architecture_diagram.md     # System architecture
    └── project-directory.md        # This file
```

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