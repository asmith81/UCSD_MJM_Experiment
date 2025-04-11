# LMM Invoice Data Extraction Comparison

## Project Overview
This project evaluates multiple open-source Large Multimodal Models (LMMs) on their ability to extract specific structured data from handwritten invoice images. The project systematically compares different models, prompts, and quantization levels to identify the optimal solution for extracting work order numbers and total costs from invoices.

### Key Features
- Evaluation of 3 open-source LMMs: Pixtral-12B, Llama-3.2-11B-Vision, and Doctr
- Field-specific extraction for work order numbers and total costs
- Multiple quantization levels (4, 8, 16, 32 bit) per model
- Systematic comparison of prompt strategies
- Comprehensive evaluation metrics and error analysis

## Project Structure
```
invoice-extraction-comparison/
├── config/                         # Configuration files
│   ├── models/                     # Model-specific configurations
│   └── prompts/                    # Field-specific prompt templates
├── data/                           # Data storage (gitignored)
├── src/                            # Source code modules
├── notebooks/                      # Jupyter notebooks
├── results/                        # Results storage (gitignored)
└── docs/                           # Documentation
```

## Setup Instructions

1. **Environment Setup**
   ```powershell
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # Install dependencies (two options available)

   # Option 1: Using environment.py script
   python src/environment.py

   # Option 2: Manual installation
   pip install -r requirements.txt
   ```

   The `environment.py` script will:
   - Set up project directory structure
   - Install PyTorch with CUDA support
   - Install remaining dependencies
   - Verify CUDA availability
   - Create necessary project directories

2. **Data Preparation**
   - Place invoice images in `data/images/`
   - Ensure ground truth data is in `data/ground_truth.csv`

3. **Configuration**
   - Review and adjust model configurations in `config/models/`
   - Customize prompts in `config/prompts/`

## Usage

1. **Model Evaluation**
   - Run model-specific notebooks in `notebooks/`:
     - `pixtral_evaluation.ipynb`
     - `llama_vision_evaluation.ipynb`
     - `doctr_evaluation.ipynb`

2. **Results Analysis**
   - Use `results_analysis.ipynb` to analyze and visualize results
   - Results are stored in `results/` directory

## Development Guidelines

- Follow functional programming approach with stateless functions
- Keep functions small and focused (< 25 lines)
- Maximum 3 levels of nesting in any function
- Document all functions with clear docstrings
- Use simple configuration files (YAML/JSON)
- Implement core error handling for critical operations

## Documentation

- `docs/project_overview.md`: Project goals and scope
- `docs/project_rules.md`: Implementation guidelines
- `docs/project_todo.md`: Task list and progress tracking
- `docs/interface_control_document.md`: Component interfaces
- `docs/project_dir.md`: Detailed directory structure

## Results

Results are organized by field type and stored in JSON format, including:
- Raw and normalized values
- Processing times
- Evaluation metrics
- Error categorization

## Contributing

1. Follow the project structure and coding guidelines
2. Document all changes
3. Update relevant documentation files
4. Test changes thoroughly

## License

[Add appropriate license information]

## Contact

[Add contact information] 