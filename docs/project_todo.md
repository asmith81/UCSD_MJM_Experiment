# Project To-Do: LMM Invoice Data Extraction Comparison (Rapid Implementation)

This to-do list is organized for the 8-hour rapid implementation timeline, with tasks grouped by functional area. Focus on completing core functionality before moving to additional features.

## Phase 1: Core Framework (2 hours)

### 1.A: Environment and Configuration (0.5 hour)
- [x] 1.1 Create `src/environment.py` for environment setup
  - Dependencies installation
  - Path configuration
  - CUDA setup for different quantization levels
- [x] 1.2 Create `src/config.py` for configuration management
  - YAML configuration loading
  - Test matrix CSV parsing
  - Result logging configuration
- [x] 1.3 Create `config/` directory structure
  - Model configuration files
  - Prompt template files
  - Test matrix CSV
- [x] 1.4 Create `src/execution.py` for test execution framework
  - Test matrix loading and filtering
  - Model loading with quantization
  - Prompt template loading
  - Result validation and logging
  - Execution tracking

### 1.B: Data Utilities (1.5 hours)
- [ ] 1.4 Create `src/data_utils.py` for data management
  - Image loading functions
  - Ground truth loading with type normalization:
    ```python
    def load_ground_truth(csv_path):
        """Load and normalize ground truth data from CSV."""
        import pandas as pd
        
        # Load CSV with proper types
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        column_mapping = {
            'Invoice': 'invoice_id',
            'Work Order Number/Numero de Orden': 'work_order_number',
            'Total': 'total_cost'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert invoice_id to string to preserve formatting
        df['invoice_id'] = df['invoice_id'].astype(str)
        
        # Clean and convert total_cost to float
        df['total_cost'] = df['total_cost'].apply(normalize_cost)
        
        # Normalize work_order_number to consistent format
        df['work_order_number'] = df['work_order_number'].apply(normalize_work_order)
        
        # Convert to dictionary for fast lookup
        result = {}
        for _, row in df.iterrows():
            result[row['invoice_id']] = {
                'work_order_number': row['work_order_number'],
                'total_cost': row['total_cost']
            }
        
        return result
        
    def normalize_cost(value):
        """Convert cost string to float, handling currency formatting."""
        if isinstance(value, (int, float)):
            return float(value)
            
        # Handle string values
        if isinstance(value, str):
            # Remove currency symbols, commas, and extra whitespace
            clean_value = value.replace('
- [ ] 1.6 Create `src/logging.py` for result tracking
  - CSV result logging functions
  - Execution tracking functions
  - Result structure definitions

## Phase 2: Model Framework (3 hours)

### 2.A: Model Infrastructure (1 hour)
- [ ] 2.1 Create `src/models/__init__.py` with model registry
  - Model type definitions
  - Factory function for model creation
- [ ] 2.2 Create `src/models/common.py` with shared functions
  - Image preprocessing
  - Model loading utilities
  - Output parsing common code

### 2.B: Model Implementations (2 hours)
- [ ] 2.3 Create `src/models/pixtral.py` for Pixtral-12B model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.4 Create `src/models/llama_vision.py` for Llama-3.2-11B-Vision model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.5 Create `src/models/doctr.py` for Doctr model
  - Model loading with quantization
  - Inference function
  - Output parsing

## Phase 3: Prompt Management (1 hour)
- [x] 3.1 Create `src/prompts.py` for prompt management
  - Prompt template loading
  - Prompt formatting for different models
  - Prompt strategy implementations (5-10 strategies)
- [x] 3.2 Create prompt template files
  - Basic extraction prompt
  - Detailed prompt
  - Few-shot example prompt
  - Step-by-step prompt
  - Locational prompt

## Phase 4: Notebook Implementation (1.5 hours)
- [x] 4.1 Create `notebooks/model_notebook_template.py`
  - Environment setup
  - Configuration loading
  - Test matrix parsing
  - Execution loop structure
  - Result logging
- [ ] 4.2 Create model-specific notebooks
  - `notebooks/pixtral_evaluation.py`
  - `notebooks/llama_vision_evaluation.py`
  - `notebooks/doctr_evaluation.py`
- [ ] 4.3 Create `notebooks/results_analysis.py`
  - Result loading functions
  - Comparative visualization
  - Best model identification

## Phase 5: Testing and Refinement (0.5 hour)
- [ ] 5.1 Test end-to-end pipeline
  - Verify model loading works
  - Confirm inference executes correctly
  - Validate result logging functions
  - Check resumability after interruption
- [ ] 5.2 Refine implementation based on initial tests
  - Fix any identified issues
  - Optimize performance bottlenecks
  - Improve error handling

## Priority Schedule (8 Hours)

| Time | Focus Area | Key Deliverables |
|------|------------|------------------|
| 0:00-0:30 | Environment & Config | Working environment setup |
| 0:30-2:00 | Data Utilities | Data loading and validation |
| 2:00-3:00 | Model Infrastructure | Model factory and registry |
| 3:00-5:00 | Model Implementations | Working model interfaces |
| 5:00-6:00 | Prompt Management | Prompt templates and formatting |
| 6:00-7:30 | Notebook Implementation | Working evaluation notebooks |
| 7:30-8:00 | Testing & Refinement | End-to-end verification |

## Implementation Notes

1. **Start with One Model**: Get one model working end-to-end before adding others
2. **Parallelize Testing**: Design for parallel execution of model notebooks
3. **Log Results Incrementally**: Save results after each test for crash resilience
4. **Focus on Functionality**: Prioritize working code over perfect architecture
5. **Documentation In Code**: Document functions as they're implemented

## Quick Checks for Each Phase

- **Phase 1**: Can we load and validate image and ground truth data?
- **Phase 2**: Can we load models at different quantization levels?
- **Phase 3**: Can we generate properly formatted prompts?
- **Phase 4**: Can we run a single test case end-to-end?
- **Phase 5**: Can we run multiple test cases and analyze results?
, '').replace(',', '').strip()
            return float(clean_value)
        
        # Default case
        return 0.0
        
    def normalize_work_order(value):
        """Normalize work order number to consistent format."""
        if pd.isna(value):
            return ""
            
        # Convert to string and strip whitespace
        return str(value).strip()
    ```
  - Field normalization utilities
  - Model output parsing with robust handling

- [ ] 1.5 Create `src/evaluation.py` for evaluation utilities
  - Result comparison with exact string matching:
    ```python
    def evaluate_extraction(prediction, ground_truth):
        """Compare extracted fields with ground truth using exact string matching."""
        # Initialize results
        results = {
            "exact_match": False,
            "work_order_match": False,
            "total_cost_match": False,
            "work_order_cer": None,
            "total_cost_cer": None,
            "error_category": None
        }
        
        # Check if required fields exist
        if not prediction or not ground_truth:
            results["error_category"] = "missing_field"
            return results
            
        # Normalize prediction field names (handle different field name variations)
        normalized_prediction = normalize_field_names(prediction)
        
        # Get values for comparison - no normalization, use as-is
        pred_wo = normalized_prediction.get('work_order_number', '')
        true_wo = ground_truth.get('work_order_number', '')
        
        pred_cost = normalized_prediction.get('total_cost', '')
        true_cost = ground_truth.get('total_cost', '')
        
        # Exact string comparison for work order
        results["work_order_match"] = (pred_wo == true_wo)
        
        # Exact string comparison for total cost
        results["total_cost_match"] = (pred_cost == true_cost)
        
        # Overall exact match
        results["exact_match"] = (
            results["work_order_match"] and 
            results["total_cost_match"]
        )
        
        # Calculate Character Error Rate and categorize errors for non-matches
        if not results["work_order_match"]:
            results["work_order_cer"] = calculate_cer(pred_wo, true_wo)
            results["error_category"] = categorize_error(pred_wo, true_wo, "work_order")
            
        if not results["total_cost_match"]:
            results["total_cost_cer"] = calculate_cer(pred_cost, true_cost)
            if not results["error_category"]:  # Don't overwrite work order error
                results["error_category"] = categorize_error(pred_cost, true_cost, "total_cost")
        
        return results
        
    def calculate_cer(pred, true):
        """Calculate Character Error Rate between predicted and true values."""
        if not true:
            return 1.0  # Maximum error if true value is empty
            
        # Levenshtein distance calculation
        distance = levenshtein_distance(pred, true)
        
        # CER = edit distance / length of reference
        return distance / len(true)
        
    def levenshtein_distance(s1, s2):
        """Calculate the edit distance between two strings."""
        # Implementation of Levenshtein distance algorithm
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
        
    def categorize_error(pred, true, field_type):
        """Categorize error type based on field and difference."""
        if not pred or not true:
            return "missing_field"
            
        if field_type == "work_order":
            # Check for transposition (same characters, different order)
            if sorted(pred) == sorted(true) and pred != true:
                return "transposition"
                
            # Check for length differences
            if len(pred) < len(true):
                return "missing_character"
                
            if len(pred) > len(true):
                return "extra_character"
                
            # Default for work order
            return "wrong_character"
            
        elif field_type == "total_cost":
            # Check for currency symbol error ($ present in one but not the other)
            if ('
- [ ] 1.6 Create `src/logging.py` for result tracking
  - CSV result logging functions
  - Execution tracking functions
  - Result structure definitions

## Phase 2: Model Framework (3 hours)

### 2.A: Model Infrastructure (1 hour)
- [ ] 2.1 Create `src/models/__init__.py` with model registry
  - Model type definitions
  - Factory function for model creation
- [ ] 2.2 Create `src/models/common.py` with shared functions
  - Image preprocessing
  - Model loading utilities
  - Output parsing common code

### 2.B: Model Implementations (2 hours)
- [ ] 2.3 Create `src/models/pixtral.py` for Pixtral-12B model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.4 Create `src/models/llama_vision.py` for Llama-3.2-11B-Vision model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.5 Create `src/models/doctr.py` for Doctr model
  - Model loading with quantization
  - Inference function
  - Output parsing

## Phase 3: Prompt Management (1 hour)
- [ ] 3.1 Create `src/prompts.py` for prompt management
  - Prompt template loading
  - Prompt formatting for different models
  - Prompt strategy implementations (5-10 strategies)
- [ ] 3.2 Create prompt template files
  - Basic extraction prompt
  - Detailed prompt
  - Few-shot example prompt
  - Step-by-step prompt
  - Locational prompt

## Phase 4: Notebook Implementation (1.5 hours)
- [ ] 4.1 Create `notebooks/model_notebook_template.ipynb`
  - Environment setup
  - Configuration loading
  - Test matrix parsing
  - Execution loop structure
  - Result logging
- [ ] 4.2 Create model-specific notebooks
  - `notebooks/pixtral_evaluation.ipynb`
  - `notebooks/llama_vision_evaluation.ipynb`
  - `notebooks/doctr_evaluation.ipynb`
- [ ] 4.3 Create `notebooks/results_analysis.ipynb`
  - Result loading functions
  - Comparative visualization
  - Best model identification

## Phase 5: Testing and Refinement (0.5 hour)
- [ ] 5.1 Test end-to-end pipeline
  - Verify model loading works
  - Confirm inference executes correctly
  - Validate result logging functions
  - Check resumability after interruption
- [ ] 5.2 Refine implementation based on initial tests
  - Fix any identified issues
  - Optimize performance bottlenecks
  - Improve error handling

## Priority Schedule (8 Hours)

| Time | Focus Area | Key Deliverables |
|------|------------|------------------|
| 0:00-0:30 | Environment & Config | Working environment setup |
| 0:30-2:00 | Data Utilities | Data loading and validation |
| 2:00-3:00 | Model Infrastructure | Model factory and registry |
| 3:00-5:00 | Model Implementations | Working model interfaces |
| 5:00-6:00 | Prompt Management | Prompt templates and formatting |
| 6:00-7:30 | Notebook Implementation | Working evaluation notebooks |
| 7:30-8:00 | Testing & Refinement | End-to-end verification |

## Implementation Notes

1. **Start with One Model**: Get one model working end-to-end before adding others
2. **Parallelize Testing**: Design for parallel execution of model notebooks
3. **Log Results Incrementally**: Save results after each test for crash resilience
4. **Focus on Functionality**: Prioritize working code over perfect architecture
5. **Documentation In Code**: Document functions as they're implemented

## Quick Checks for Each Phase

- **Phase 1**: Can we load and validate image and ground truth data?
- **Phase 2**: Can we load models at different quantization levels?
- **Phase 3**: Can we generate properly formatted prompts?
- **Phase 4**: Can we run a single test case end-to-end?
- **Phase 5**: Can we run multiple test cases and analyze results?
 in pred) != ('
- [ ] 1.6 Create `src/logging.py` for result tracking
  - CSV result logging functions
  - Execution tracking functions
  - Result structure definitions

## Phase 2: Model Framework (3 hours)

### 2.A: Model Infrastructure (1 hour)
- [ ] 2.1 Create `src/models/__init__.py` with model registry
  - Model type definitions
  - Factory function for model creation
- [ ] 2.2 Create `src/models/common.py` with shared functions
  - Image preprocessing
  - Model loading utilities
  - Output parsing common code

### 2.B: Model Implementations (2 hours)
- [ ] 2.3 Create `src/models/pixtral.py` for Pixtral-12B model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.4 Create `src/models/llama_vision.py` for Llama-3.2-11B-Vision model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.5 Create `src/models/doctr.py` for Doctr model
  - Model loading with quantization
  - Inference function
  - Output parsing

## Phase 3: Prompt Management (1 hour)
- [ ] 3.1 Create `src/prompts.py` for prompt management
  - Prompt template loading
  - Prompt formatting for different models
  - Prompt strategy implementations (5-10 strategies)
- [ ] 3.2 Create prompt template files
  - Basic extraction prompt
  - Detailed prompt
  - Few-shot example prompt
  - Step-by-step prompt
  - Locational prompt

## Phase 4: Notebook Implementation (1.5 hours)
- [ ] 4.1 Create `notebooks/model_notebook_template.ipynb`
  - Environment setup
  - Configuration loading
  - Test matrix parsing
  - Execution loop structure
  - Result logging
- [ ] 4.2 Create model-specific notebooks
  - `notebooks/pixtral_evaluation.ipynb`
  - `notebooks/llama_vision_evaluation.ipynb`
  - `notebooks/doctr_evaluation.ipynb`
- [ ] 4.3 Create `notebooks/results_analysis.ipynb`
  - Result loading functions
  - Comparative visualization
  - Best model identification

## Phase 5: Testing and Refinement (0.5 hour)
- [ ] 5.1 Test end-to-end pipeline
  - Verify model loading works
  - Confirm inference executes correctly
  - Validate result logging functions
  - Check resumability after interruption
- [ ] 5.2 Refine implementation based on initial tests
  - Fix any identified issues
  - Optimize performance bottlenecks
  - Improve error handling

## Priority Schedule (8 Hours)

| Time | Focus Area | Key Deliverables |
|------|------------|------------------|
| 0:00-0:30 | Environment & Config | Working environment setup |
| 0:30-2:00 | Data Utilities | Data loading and validation |
| 2:00-3:00 | Model Infrastructure | Model factory and registry |
| 3:00-5:00 | Model Implementations | Working model interfaces |
| 5:00-6:00 | Prompt Management | Prompt templates and formatting |
| 6:00-7:30 | Notebook Implementation | Working evaluation notebooks |
| 7:30-8:00 | Testing & Refinement | End-to-end verification |

## Implementation Notes

1. **Start with One Model**: Get one model working end-to-end before adding others
2. **Parallelize Testing**: Design for parallel execution of model notebooks
3. **Log Results Incrementally**: Save results after each test for crash resilience
4. **Focus on Functionality**: Prioritize working code over perfect architecture
5. **Documentation In Code**: Document functions as they're implemented

## Quick Checks for Each Phase

- **Phase 1**: Can we load and validate image and ground truth data?
- **Phase 2**: Can we load models at different quantization levels?
- **Phase 3**: Can we generate properly formatted prompts?
- **Phase 4**: Can we run a single test case end-to-end?
- **Phase 5**: Can we run multiple test cases and analyze results?
 in true):
                return "currency_error"
                
            # Check for decimal point errors
            if ('.' in pred) != ('.' in true):
                return "decimal_error"
                
            # Check for formatting errors (commas, spaces)
            if any(c in pred for c in ',.') != any(c in true for c in ',.'):
                return "formatting_error"
                
            # Default for cost
            return "digit_error"
            
        return "unknown_error"
    ```
  - Character Error Rate (CER) calculation
  - Error categorization functions
  - Detailed result logging
- [ ] 1.6 Create `src/logging.py` for result tracking
  - CSV result logging functions
  - Execution tracking functions
  - Result structure definitions

## Phase 2: Model Framework (3 hours)

### 2.A: Model Infrastructure (1 hour)
- [ ] 2.1 Create `src/models/__init__.py` with model registry
  - Model type definitions
  - Factory function for model creation
- [ ] 2.2 Create `src/models/common.py` with shared functions
  - Image preprocessing
  - Model loading utilities
  - Output parsing common code

### 2.B: Model Implementations (2 hours)
- [ ] 2.3 Create `src/models/pixtral.py` for Pixtral-12B model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.4 Create `src/models/llama_vision.py` for Llama-3.2-11B-Vision model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [ ] 2.5 Create `src/models/doctr.py` for Doctr model
  - Model loading with quantization
  - Inference function
  - Output parsing

## Phase 3: Prompt Management (1 hour)
- [ ] 3.1 Create `src/prompts.py` for prompt management
  - Prompt template loading
  - Prompt formatting for different models
  - Prompt strategy implementations (5-10 strategies)
- [ ] 3.2 Create prompt template files
  - Basic extraction prompt
  - Detailed prompt
  - Few-shot example prompt
  - Step-by-step prompt
  - Locational prompt

## Phase 4: Notebook Implementation (1.5 hours)
- [ ] 4.1 Create `notebooks/model_notebook_template.ipynb`
  - Environment setup
  - Configuration loading
  - Test matrix parsing
  - Execution loop structure
  - Result logging
- [ ] 4.2 Create model-specific notebooks
  - `notebooks/pixtral_evaluation.ipynb`
  - `notebooks/llama_vision_evaluation.ipynb`
  - `notebooks/doctr_evaluation.ipynb`
- [ ] 4.3 Create `notebooks/results_analysis.ipynb`
  - Result loading functions
  - Comparative visualization
  - Best model identification

## Phase 5: Testing and Refinement (0.5 hour)
- [ ] 5.1 Test end-to-end pipeline
  - Verify model loading works
  - Confirm inference executes correctly
  - Validate result logging functions
  - Check resumability after interruption
- [ ] 5.2 Refine implementation based on initial tests
  - Fix any identified issues
  - Optimize performance bottlenecks
  - Improve error handling

## Priority Schedule (8 Hours)

| Time | Focus Area | Key Deliverables |
|------|------------|------------------|
| 0:00-0:30 | Environment & Config | Working environment setup |
| 0:30-2:00 | Data Utilities | Data loading and validation |
| 2:00-3:00 | Model Infrastructure | Model factory and registry |
| 3:00-5:00 | Model Implementations | Working model interfaces |
| 5:00-6:00 | Prompt Management | Prompt templates and formatting |
| 6:00-7:30 | Notebook Implementation | Working evaluation notebooks |
| 7:30-8:00 | Testing & Refinement | End-to-end verification |

## Implementation Notes

1. **Start with One Model**: Get one model working end-to-end before adding others
2. **Parallelize Testing**: Design for parallel execution of model notebooks
3. **Log Results Incrementally**: Save results after each test for crash resilience
4. **Focus on Functionality**: Prioritize working code over perfect architecture
5. **Documentation In Code**: Document functions as they're implemented

## Quick Checks for Each Phase

- **Phase 1**: Can we load and validate image and ground truth data?
- **Phase 2**: Can we load models at different quantization levels?
- **Phase 3**: Can we generate properly formatted prompts?
- **Phase 4**: Can we run a single test case end-to-end?
- **Phase 5**: Can we run multiple test cases and analyze results?