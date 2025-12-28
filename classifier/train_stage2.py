#!/usr/bin/env python3
"""
Training script for Stage 2 jailbreak classifier.
"""

import argparse
import yaml
import logging
from pathlib import Path
from data_loader import DataLoader
from stage2_classifier import Stage2Classifier

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train_stage2.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 jailbreak classifier")
    parser.add_argument("--data", required=True, help="Path to stage2_data.csv")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(config.get('general', {}).get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Stage 2 training")
    
    try:
        # Load data
        data_loader = DataLoader(random_seed=config.get('general', {}).get('random_seed', 42))
        prompts, types = data_loader.load_stage2_data(args.data)
        
        # Initialize classifier
        classifier = Stage2Classifier({**config['stage2'], **config['general']})
        classifier.load_embedding_model()
        
        # Train models
        classifier.train_models(prompts, types)
        classifier.save_models("models")
        
        logger.info("Stage 2 training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()