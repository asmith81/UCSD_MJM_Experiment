"""
Data management utilities for LMM invoice data extraction comparison.
Handles image loading and ground truth data management with type normalization.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load and validate an invoice image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is invalid
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def normalize_work_order(value: str) -> str:
    """
    Normalize work order number to consistent format.
    
    Args:
        value: Raw work order number
        
    Returns:
        Normalized work order number
    """
    if pd.isna(value):
        return ""
        
    # Convert to string and strip whitespace
    return str(value).strip()

def normalize_total_cost(value: Union[str, float]) -> float:
    """
    Normalize total cost to float value.
    
    Args:
        value: Raw total cost value
        
    Returns:
        Normalized float value
        
    Raises:
        ValueError: If value cannot be normalized
    """
    if pd.isna(value):
        return 0.0
        
    # Convert to string for processing
    str_value = str(value)
    
    # Remove currency symbols and whitespace
    clean_value = str_value.replace('$', '').strip()
    
    # Remove commas and convert to float
    try:
        return float(clean_value.replace(',', ''))
    except ValueError:
        logger.error(f"Could not normalize total cost value: {value}")
        return 0.0

def load_ground_truth(csv_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load and normalize ground truth data from CSV.
    
    Args:
        csv_path: Path to ground truth CSV file
        
    Returns:
        Dictionary mapping image IDs to normalized ground truth values
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
            
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['image_id', 'work_order_number', 'total_cost']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
            
        # Initialize results dictionary
        ground_truth = {}
        
        # Process each row
        for _, row in df.iterrows():
            image_id = str(row['image_id'])
            
            # Normalize values
            work_order = normalize_work_order(row['work_order_number'])
            total_cost = normalize_total_cost(row['total_cost'])
            
            # Store normalized values
            ground_truth[image_id] = {
                'work_order_number': {
                    'raw_value': str(row['work_order_number']),
                    'normalized_value': work_order
                },
                'total_cost': {
                    'raw_value': str(row['total_cost']),
                    'normalized_value': total_cost
                }
            }
            
        return ground_truth
        
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        raise

def validate_image_directory(image_dir: Union[str, Path]) -> List[str]:
    """
    Validate image directory and return list of valid image files.
    
    Args:
        image_dir: Path to image directory
        
    Returns:
        List of valid image file paths
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid images found
    """
    try:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        # Get all image files
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for file in image_dir.glob('*'):
            if file.suffix.lower() in valid_extensions:
                image_files.append(str(file))
                
        if not image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")
            
        return image_files
        
    except Exception as e:
        logger.error(f"Error validating image directory: {str(e)}")
        raise 