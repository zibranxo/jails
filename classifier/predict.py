#!/usr/bin/env python3
"""
Enhanced CLI tool for end-to-end jailbreak detection with comprehensive evaluation.
"""

import argparse
import yaml
import pandas as pd
import logging
from pathlib import Path
from stage1_detector import Stage1Detector
from stage2_classifier import Stage2Classifier
from evaluate_predictions import JailbreakEvaluator, save_detailed_metrics

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/predict.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run jailbreak detection inference with evaluation")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ground-truth", help="Path to ground truth CSV for evaluation")
    parser.add_argument("--metrics-output", help="Path to save detailed metrics JSON")
    parser.add_argument("--report-output", help="Path to save evaluation report")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(config.get('general', {}).get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting inference with evaluation")
    
    try:
        # Load input data
        df = pd.read_csv(args.input)
        if 'prompt' not in df.columns:
            raise ValueError("Input CSV must have 'prompt' column")
        
        prompts = df['prompt'].tolist()
        logger.info(f"Processing {len(prompts)} prompts")
        
        # Initialize models
        stage1_detector = Stage1Detector(config['stage1'])
        stage1_detector.load_embedding_model()
        stage1_detector.load_index(config['stage1']['faiss_index_path'])
        
        stage2_classifier = Stage2Classifier({**config['stage2'], **config['general']})
        stage2_classifier.load_embedding_model()
        stage2_classifier.load_models("models")
        
        # Run inference
        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Stage 1: Binary detection
            is_jailbreak = stage1_detector.detect_jailbreak(prompt)
            stage1_label = 1 if is_jailbreak else 0
            
            # Stage 2: Type classification (only if jailbreak detected)
            if is_jailbreak:
                jailbreak_type, confidence = stage2_classifier.classify_jailbreak(prompt)
            else:
                jailbreak_type, confidence = "", ""
            
            results.append({
                'prompt': prompt,
                'stage1_label': stage1_label,
                'stage2_type': jailbreak_type,
                'stage2_confidence': confidence
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        
        # Log summary
        jailbreaks_detected = sum(1 for r in results if r['stage1_label'] == 1)
        logger.info(f"Inference completed: {jailbreaks_detected}/{len(prompts)} jailbreaks detected")
        logger.info(f"Results saved to {args.output}")
        
        # Evaluation (if ground truth provided)
        if args.ground_truth:
            logger.info("Running evaluation against ground truth")
            
            # Load ground truth
            gt_df = pd.read_csv(args.ground_truth)
            required_cols = ['prompt', 'stage1_label']
            if not all(col in gt_df.columns for col in required_cols):
                raise ValueError(f"Ground truth must have columns: {required_cols}")
            
            # Run evaluation
            evaluator = JailbreakEvaluator()
            eval_results = evaluator.evaluate_end_to_end(gt_df, results_df)
            
            # Save detailed metrics
            if args.metrics_output:
                save_detailed_metrics(eval_results, args.metrics_output)
            
            # Generate and save report
            report = evaluator.generate_report(args.report_output)
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(report)
            
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
