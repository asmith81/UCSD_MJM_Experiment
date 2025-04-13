"""
Prompt template management module.

This module provides functionality for loading and managing prompt templates
from YAML configuration files.
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_prompt_template(prompt_strategy: str) -> str:
    """
    Load a prompt template for a specific strategy.
    
    Args:
        prompt_strategy: Name of the prompt strategy (e.g., 'basic_extraction', 'detailed')
        
    Returns:
        Loaded prompt template string
        
    Raises:
        FileNotFoundError: If prompt template doesn't exist
        ValueError: If prompt template is invalid
    """
    try:
        # Construct path to prompt template
        prompt_path = Path("config/prompts") / f"{prompt_strategy}.yaml"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            
        # Load and parse YAML
        with open(prompt_path, 'r') as f:
            template_data = yaml.safe_load(f)
            
        # Validate template structure
        if 'template' not in template_data:
            raise ValueError(f"Invalid prompt template: missing 'template' field in {prompt_path}")
            
        return template_data['template']
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing prompt template {prompt_strategy}: {str(e)}")
        raise ValueError(f"Invalid YAML in prompt template {prompt_strategy}")
    except Exception as e:
        logger.error(f"Error loading prompt template {prompt_strategy}: {str(e)}")
        raise 