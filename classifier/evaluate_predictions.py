"""
Comprehensive evaluation module for jailbreak detection system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class JailbreakEvaluator:
    """Comprehensive evaluation for jailbreak detection system."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_stage1_binary(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """
        Evaluate Stage 1 binary classification performance.
        
        Args:
            y_true: Ground truth binary labels (0=safe, 1=jailbreak)
            y_pred: Predicted binary labels (0=safe, 1=jailbreak)
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info("Evaluating Stage 1 binary classification")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_binary = f1_score(y_true, y_pred, average='binary')
        
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        precision_binary = precision_score(y_true, y_pred, average='binary')
        
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        recall_binary = recall_score(y_true, y_pred, average='binary')
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity and other derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_binary': f1_binary,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_binary': precision_binary,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'recall_binary': recall_binary,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'matthews_corrcoef': mcc,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'classification_report': class_report
        }
        
        self.results['stage1_binary'] = results
        return results
    
    def evaluate_stage2_multiclass(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """
        Evaluate Stage 2 multi-class classification performance.
        
        Args:
            y_true: Ground truth jailbreak types
            y_pred: Predicted jailbreak types
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info("Evaluating Stage 2 multi-class classification")
        
        # Handle empty predictions (when Stage 1 predicts safe)
        valid_indices = [(i, true, pred) for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                        if true != '' and pred != '']
        
        if not valid_indices:
            logger.warning("No valid predictions for Stage 2 evaluation")
            return {'error': 'No valid predictions found'}
        
        _, valid_y_true, valid_y_pred = zip(*valid_indices)
        
        # Basic metrics
        accuracy = accuracy_score(valid_y_true, valid_y_pred)
        f1_macro = f1_score(valid_y_true, valid_y_pred, average='macro')
        f1_weighted = f1_score(valid_y_true, valid_y_pred, average='weighted')
        f1_micro = f1_score(valid_y_true, valid_y_pred, average='micro')
        
        precision_macro = precision_score(valid_y_true, valid_y_pred, average='macro')
        precision_weighted = precision_score(valid_y_true, valid_y_pred, average='weighted')
        precision_micro = precision_score(valid_y_true, valid_y_pred, average='micro')
        
        recall_macro = recall_score(valid_y_true, valid_y_pred, average='macro')
        recall_weighted = recall_score(valid_y_true, valid_y_pred, average='weighted')
        recall_micro = recall_score(valid_y_true, valid_y_pred, average='micro')
        
        # Classification report with per-class metrics
        class_report = classification_report(valid_y_true, valid_y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(valid_y_true, valid_y_pred)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'recall_micro': recall_micro,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'num_classes': len(set(valid_y_true)),
            'total_predictions': len(valid_y_pred)
        }
        
        self.results['stage2_multiclass'] = results
        return results
    
    def evaluate_end_to_end(self, df_true: pd.DataFrame, df_pred: pd.DataFrame) -> Dict:
        """
        Evaluate complete end-to-end system performance.
        
        Args:
            df_true: DataFrame with ground truth (prompt, stage1_label, stage2_type)
            df_pred: DataFrame with predictions (prompt, stage1_label, stage2_type, stage2_confidence)
            
        Returns:
            Dictionary containing comprehensive metrics
        """
        logger.info("Evaluating end-to-end system performance")
        
        # Stage 1 evaluation
        stage1_metrics = self.evaluate_stage1_binary(
            df_true['stage1_label'].tolist(),
            df_pred['stage1_label'].tolist()
        )
        
        # Stage 2 evaluation (only for jailbreak cases)
        jailbreak_mask = df_true['stage1_label'] == 1
        if jailbreak_mask.sum() > 0:
            stage2_metrics = self.evaluate_stage2_multiclass(
                df_true.loc[jailbreak_mask, 'stage2_type'].tolist(),
                df_pred.loc[jailbreak_mask, 'stage2_type'].tolist()
            )
        else:
            stage2_metrics = {'error': 'No jailbreak samples for Stage 2 evaluation'}
        
        # Combined metrics
        results = {
            'stage1_binary': stage1_metrics,
            'stage2_multiclass': stage2_metrics,
            'total_samples': len(df_true),
            'jailbreak_samples': jailbreak_mask.sum(),
            'safe_samples': (~jailbreak_mask).sum()
        }
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        report_lines.append("# Jailbreak Detection System Evaluation Report\n")
        
        if 'stage1_binary' in self.results:
            s1 = self.results['stage1_binary']
            report_lines.extend([
                "## Stage 1: Binary Classification Metrics",
                f"- **Accuracy**: {s1['accuracy']:.4f}",
                f"- **F1-Score (Binary)**: {s1['f1_binary']:.4f}",
                f"- **F1-Score (Macro)**: {s1['f1_macro']:.4f}",
                f"- **F1-Score (Weighted)**: {s1['f1_weighted']:.4f}",
                f"- **Precision (Binary)**: {s1['precision_binary']:.4f}",
                f"- **Recall (Binary)**: {s1['recall_binary']:.4f}",
                f"- **Specificity**: {s1['specificity']:.4f}",
                f"- **Matthews Correlation Coefficient**: {s1['matthews_corrcoef']:.4f}",
                "",
                "### Confusion Matrix",
                f"- True Positives: {s1['true_positives']}",
                f"- True Negatives: {s1['true_negatives']}",
                f"- False Positives: {s1['false_positives']}",
                f"- False Negatives: {s1['false_negatives']}",
                ""
            ])
        
        if 'stage2_multiclass' in self.results and 'error' not in self.results['stage2_multiclass']:
            s2 = self.results['stage2_multiclass']
            report_lines.extend([
                "## Stage 2: Multi-class Classification Metrics",
                f"- **Accuracy**: {s2['accuracy']:.4f}",
                f"- **F1-Score (Macro)**: {s2['f1_macro']:.4f}",
                f"- **F1-Score (Weighted)**: {s2['f1_weighted']:.4f}",
                f"- **F1-Score (Micro)**: {s2['f1_micro']:.4f}",
                f"- **Precision (Macro)**: {s2['precision_macro']:.4f}",
                f"- **Recall (Macro)**: {s2['recall_macro']:.4f}",
                f"- **Number of Classes**: {s2['num_classes']}",
                f"- **Total Predictions**: {s2['total_predictions']}",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text

def save_detailed_metrics(results: Dict, output_path: str) -> None:
    """Save detailed metrics to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Detailed metrics saved to {output_path}")
