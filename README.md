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
│   ├── environment.py              # Environment setup and paths
│   ├── config.py                   # Configuration management
│   └── models/                     # Model implementations
├── notebooks/                      # Jupyter notebooks
├── results/                        # Results storage (gitignored)
└── docs/                           # Documentation
    ├── adr/                        # Architecture Decision Records
    ├── project_overview.md         # Project goals and scope
    ├── project_rules.md            # Implementation guidelines
    ├── project_todo.md             # Task list and progress tracking
    ├── interface_control_document.md # Component interfaces
    └── project_dir.md              # Detailed directory structure
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
   - Configuration is managed through `src/config.py` with dependency injection pattern

## Usage

1. **Model Evaluation**
   ```python
   # In notebook
   from src.environment import setup_environment
   from src.config import load_yaml_config, setup_logging_config

   # Setup environment once at notebook start
   env = setup_environment()
   paths = env['paths']

   # Load and setup config
   config = load_yaml_config('config/config.yaml')
   log_config = setup_logging_config(config, paths['logs'])
   ```

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
- Follow dependency injection pattern for configuration management

## Documentation

- `docs/adr/`: Architecture Decision Records
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

## Model Configurations

### Pixtral-12B
- Uses direct prompt format for simplicity
- Optimized for single-image extraction
- No conversation history needed
- Simple prompt structure

### Llama-3.2-11B-Vision
- Direct prompt format with system message
- Optimized for invoice extraction
- Supports LoRA fine-tuning
- Image preprocessing requirements:
  - Max size: 1120x1120
  - RGB conversion
  - Normalization
  - Aspect ratio maintenance
- Higher token limit (2048) for detailed responses
- No content safety checks (business documents only)

### Doctr
- Two-stage architecture:
  - Text detection (DBNet)
  - Text recognition (CRNN)
- KIE support for structured extraction
- PyTorch backend with visualization support
- Document handling:
  - PDF support
  - Single and multi-page images
  - Preprocessed images only
- Output structure:
  - JSON format
  - Confidence scores
  - Geometry information
- Lower hardware requirements (8GB GPU)
- Higher batch processing capability

## Configuration Management
- YAML-based configuration files
- Clear documentation of decisions
- Environment setup in source modules
- Model-specific loading and inference functions 