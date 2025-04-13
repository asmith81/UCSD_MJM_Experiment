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
    config_dir: Path
    prompts_dir: Path
    cuda_available: bool
    cuda_version: Optional[str]
    dependencies_installed: bool
    environment: str  # 'local' or 'runpod'

def detect_environment() -> str:
    """
    Detect the current environment.
    
    Returns:
        str: 'local' or 'runpod'
    """
    # Check for RunPod specific environment variables
    if os.getenv('RUNPOD_POD_ID') is not None:
        return 'runpod'
    
    # Check for CUDA availability and GPU memory
    if torch.cuda.is_available():
        try:
            # RunPod containers typically have large GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 20 * 1024 * 1024 * 1024:  # More than 20GB
                return 'runpod'
        except:
            pass
    
    return 'local'

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
        'logs_dir': project_root / 'logs',
        'config_dir': project_root / 'config',
        'prompts_dir': project_root / 'config' / 'prompts'
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

def download_model(model_name: str, model_path: Path, repo_id: str) -> bool:
    """
    Download model from HuggingFace.
    
    Args:
        model_name: Name of the model
        model_path: Path where model should be downloaded
        repo_id: HuggingFace repository ID
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        RuntimeError: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
        
        if model_path.exists():
            logging.info(f"Model {model_name} already exists at {model_path}")
            return True
            
        logging.info(f"Downloading model {model_name} from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        return True
        
    except Exception as e:
        logging.error(f"Failed to download model {model_name}: {str(e)}")
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
        
        # Detect environment
        env = detect_environment()
        
        # Combine all configurations
        env_config: EnvironmentConfig = {
            **paths,
            'cuda_available': cuda_info['cuda_available'],
            'cuda_version': cuda_info['cuda_version'],
            'dependencies_installed': dependencies_installed,
            'environment': env
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
    env = detect_environment()
    overrides = {
        'local': {
            'log_level': 'DEBUG',
            'use_gpu': False,
            'prompt_cache': False
        },
        'runpod': {
            'log_level': 'INFO',
            'use_gpu': True,
            'prompt_cache': True
        }
    }
    return overrides.get(env, {})

if __name__ == '__main__':
    # Run environment setup when script is executed directly
    env_config = setup_environment(Path(__file__).parent.parent, Path(__file__).parent.parent / 'requirements.txt')
    logger = logging.getLogger(__name__)
    logger.info("Environment setup completed")
    logger.info(f"Environment: {env_config['environment']}")
    logger.info(f"CUDA available: {env_config['cuda_available']}")
    logger.info(f"Project paths: {env_config}") 