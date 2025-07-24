"""
Data loading and preprocessing utilities for jailbreak detection system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of training data."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize DataLoader with random seed for reproducibility.
        
        Args:
            random_seed: Random seed for numpy operations
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_stage1_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """
        Load Stage 1 binary classification data.
        
        Args:
            data_path: Path to CSV file with 'prompt' and 'unsafe' columns
            
        Returns:
            Tuple of (prompts, labels) where labels are 0/1 integers
        """
        logger.info(f"Loading Stage 1 data from {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Validate required columns
        required_cols = ['prompt', 'unsafe']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and validate data
        df = df.dropna(subset=required_cols)
        df['unsafe'] = df['unsafe'].astype(int)
        
        # Validate binary labels
        if not df['unsafe'].isin([0, 1]).all():
            raise ValueError("'unsafe' column must contain only 0 and 1 values")
        
        logger.info(f"Loaded {len(df)} samples: {df['unsafe'].sum()} unsafe, {(df['unsafe']==0).sum()} safe")
        
        return df['prompt'].tolist(), df['unsafe'].tolist()
    
    def load_stage2_data(self, data_path: str) -> Tuple[List[str], List[str]]:
        """
        Load Stage 2 multi-class classification data.
        
        Args:
            data_path: Path to CSV file with 'prompt' and 'type' columns
            
        Returns:
            Tuple of (prompts, types)
        """
        logger.info(f"Loading Stage 2 data from {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Validate required columns
        required_cols = ['prompt', 'type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        df = df.dropna(subset=required_cols)
        
        # Get unique types
        unique_types = df['type'].unique().tolist()
        logger.info(f"Loaded {len(df)} samples with {len(unique_types)} unique types: {unique_types}")
        
        return df['prompt'].tolist(), df['type'].tolist()
    
    def get_unsafe_prompts(self, prompts: List[str], labels: List[int]) -> List[str]:
        """
        Filter prompts to return only unsafe ones (label=1).
        
        Args:
            prompts: List of prompt strings
            labels: List of binary labels (0/1)
            
        Returns:
            List of unsafe prompts
        """
        unsafe_prompts = [prompt for prompt, label in zip(prompts, labels) if label == 1]
        logger.info(f"Extracted {len(unsafe_prompts)} unsafe prompts from {len(prompts)} total")
        return unsafe_prompts
