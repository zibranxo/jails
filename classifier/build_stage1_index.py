#!/usr/bin/env python3
"""
Build Stage 1 FAISS index from harmful prompts dataset.
"""

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from stage1_detector import Stage1Detector

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/build_stage1_index.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Build Stage 1 FAISS index")
    parser.add_argument("--data", required=True, help="Path to stage1_data.csv")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(config.get('general', {}).get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Stage 1 index building")
    
    try:
        # Load harmful prompts data
        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        
        # Validate CSV format
        if 'prompt' not in df.columns or 'unsafe' not in df.columns:
            raise ValueError("CSV must have 'prompt' and 'unsafe' columns")
        
        # Extract only harmful prompts (unsafe=1)
        harmful_prompts = df[df['unsafe'] == 1]['prompt'].tolist()
        
        if not harmful_prompts:
            raise ValueError("No harmful prompts found (unsafe=1) in the dataset")
        
        logger.info(f"Found {len(harmful_prompts)} harmful prompts for indexing")
        
        # Initialize Stage 1 detector2
        detector = Stage1Detector(config['stage1'])
        detector.load_embedding_model()
        
        # Build and save FAISS index
        detector.build_faiss_index(harmful_prompts)
        detector.save_index(config['stage1']['faiss_index_path'])
        
        logger.info("Stage 1 index building completed successfully")
        logger.info(f"Index saved to: {config['stage1']['faiss_index_path']}")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise

if __name__ == "__main__":
    main()
