"""
Environment setup and configuration for the LMM Invoice Data Extraction project.
Handles dependency installation, path configuration, and CUDA setup.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_paths() -> Dict[str, Path]:
    """
    Set up and return project directory paths.
    
    Returns:
        Dict[str, Path]: Dictionary of project paths
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define project paths
    paths = {
        'root': project_root,
        'src': project_root / 'src',
        'data': project_root / 'data',
        'config': project_root / 'config',
        'notebooks': project_root / 'notebooks',
        'results': project_root / 'results',
        'logs': project_root / 'logs'
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def check_cuda_availability() -> Dict[str, bool]:
    """
    Check CUDA availability and return configuration status.
    
    Returns:
        Dict[str, bool]: Dictionary of CUDA configuration status
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        return {
            'cuda_available': cuda_available,
            'cuda_version': cuda_version,
            'device_count': torch.cuda.device_count() if cuda_available else 0
        }
    except ImportError:
        logger.warning("PyTorch not installed. CUDA check skipped.")
        return {
            'cuda_available': False,
            'cuda_version': None,
            'device_count': 0
        }

def install_dependencies(requirements_path: Optional[Path] = None) -> bool:
    """
    Install project dependencies from requirements.txt.
    
    Args:
        requirements_path: Path to requirements.txt file
        
    Returns:
        bool: True if installation was successful
    """
    if requirements_path is None:
        requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    
    if not requirements_path.exists():
        logger.error(f"Requirements file not found at {requirements_path}")
        return False
    
    try:
        # Install PyTorch with CUDA support first
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'torch==2.2.0', 'torchvision==0.17.0',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ], check=True)
        
        # Install remaining requirements
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '-r', str(requirements_path)
        ], check=True)
        
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment() -> Dict:
    """
    Set up the complete environment including paths, CUDA, and dependencies.
    
    Returns:
        Dict: Dictionary containing environment configuration
    """
    # Set up paths
    paths = setup_paths()
    
    # Check CUDA availability
    cuda_config = check_cuda_availability()
    
    # Install dependencies
    dependencies_installed = install_dependencies()
    
    return {
        'paths': paths,
        'cuda_config': cuda_config,
        'dependencies_installed': dependencies_installed
    }

if __name__ == '__main__':
    # Run environment setup when script is executed directly
    env_config = setup_environment()
    logger.info("Environment setup completed")
    logger.info(f"CUDA available: {env_config['cuda_config']['cuda_available']}")
    logger.info(f"Project paths: {env_config['paths']}") 