"""
Environment setup and dependency management for LMM invoice data extraction comparison.
Handles path setup, CUDA configuration, and dependency installation.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TypedDict
import torch

class EnvironmentConfig(TypedDict):
    project_root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path
    logs_dir: Path
    cuda_available: bool
    cuda_version: Optional[str]
    dependencies_installed: bool

def setup_paths(project_root: Path) -> Dict[str, Path]:
    """
    Setup project directory structure.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary of project paths
        
    Raises:
        OSError: If directory creation fails
    """
    paths = {
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'models_dir': project_root / 'models',
        'results_dir': project_root / 'results',
        'logs_dir': project_root / 'logs'
    }
    
    try:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create project directories: {str(e)}")
    
    return paths

def check_cuda_availability() -> Dict[str, Any]:
    """
    Check CUDA availability and version.
    
    Returns:
        Dictionary containing CUDA status and version
    """
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    
    return {
        'cuda_available': cuda_available,
        'cuda_version': cuda_version
    }

def install_dependencies(requirements_path: Path) -> bool:
    """
    Install project dependencies from requirements.txt.
    
    Args:
        requirements_path: Path to requirements.txt file
        
    Returns:
        bool: True if installation successful, False otherwise
        
    Raises:
        FileNotFoundError: If requirements.txt not found
        subprocess.CalledProcessError: If pip install fails
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)])
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {str(e)}")
        return False

def setup_environment(project_root: Path, requirements_path: Path) -> EnvironmentConfig:
    """
    Setup complete project environment.
    
    Args:
        project_root: Root directory of the project
        requirements_path: Path to requirements.txt file
        
    Returns:
        EnvironmentConfig: Complete environment configuration
        
    Raises:
        OSError: If environment setup fails
    """
    try:
        # Setup paths
        paths = setup_paths(project_root)
        
        # Check CUDA
        cuda_info = check_cuda_availability()
        
        # Install dependencies
        dependencies_installed = install_dependencies(requirements_path)
        
        # Combine all configurations
        env_config: EnvironmentConfig = {
            **paths,
            'cuda_available': cuda_info['cuda_available'],
            'cuda_version': cuda_info['cuda_version'],
            'dependencies_installed': dependencies_installed
        }
        
        return env_config
    except Exception as e:
        raise OSError(f"Environment setup failed: {str(e)}")

def get_environment_overrides() -> Dict[str, Any]:
    """
    Get environment-specific configuration overrides.
    
    Returns:
        Dictionary of environment-specific overrides
    """
    env = os.getenv('ENVIRONMENT', 'development')
    overrides = {
        'development': {
            'log_level': 'DEBUG',
            'use_gpu': False
        },
        'production': {
            'log_level': 'INFO',
            'use_gpu': True
        }
    }
    return overrides.get(env, {})

if __name__ == '__main__':
    # Run environment setup when script is executed directly
    env_config = setup_environment(Path(__file__).parent.parent, Path(__file__).parent.parent / 'requirements.txt')
    logger.info("Environment setup completed")
    logger.info(f"CUDA available: {env_config['cuda_available']}")
    logger.info(f"Project paths: {env_config}") 