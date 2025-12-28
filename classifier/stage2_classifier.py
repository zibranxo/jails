"""
Stage 2: Fine-grained jailbreak type classification using ensemble methods.
"""

import numpy as np
import logging
import joblib
from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Stage2Classifier:
    """Fine-grained jailbreak type classifier using ensemble methods."""
    
    def __init__(self, config: dict):
        """
        Initialize Stage 2 classifier.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.embedding_model = None
        self.pca_transformer = None
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.xgb_model = None
        self.n_components = config.get('pca_components', 50)
        
        # Set random seeds for reproducibility[7][8]
        self.random_seed = config.get('random_state', 42)
        np.random.seed(self.random_seed)
    
    def load_embedding_model(self) -> None:
        """Load the sentence transformer model for embeddings."""
        model_name = self.config.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def extract_features(self, prompts: List[str]) -> np.ndarray:
        """
        Extract and optionally reduce dimensionality of embeddings.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        logger.info(f"Extracting features from {len(prompts)} prompts")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(prompts, show_progress_bar=True)
        embeddings = np.array(embeddings)
        
        # Apply PCA if transformer exists[10][16]
        if self.pca_transformer is not None:
            embeddings = self.pca_transformer.transform(embeddings)
            logger.debug(f"Applied PCA transformation: {embeddings.shape}")
        
        return embeddings
    
    def fit_pca(self, embeddings: np.ndarray) -> None:
        """
        Fit PCA transformer on embeddings.
        
        Args:
            embeddings: Input embeddings matrix
        """
        logger.info(f"Fitting PCA with {self.n_components} components")
        self.pca_transformer = PCA(n_components=self.n_components, random_state=self.random_seed)
        self.pca_transformer.fit(embeddings)
        
        explained_variance = self.pca_transformer.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance ratio: {explained_variance:.3f}")
    
    def train_models(self, prompts: List[str], types: List[str]) -> None:
        """
        Train ensemble models on prompt data.
        
        Args:
            prompts: List of prompt strings
            types: List of jailbreak type labels
        """
        logger.info(f"Training models on {len(prompts)} samples")
        
        # Extract features
        X = self.extract_features(prompts)
        
        # If PCA not fitted, fit it now
        if self.pca_transformer is None:
            self.fit_pca(X)
            X = self.pca_transformer.transform(X)
        
        # Encode labels
        y = self.label_encoder.fit_transform(types)
        n_classes = len(self.label_encoder.classes_)
        logger.info(f"Training on {n_classes} classes: {list(self.label_encoder.classes_)}")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        
        # Train Random Forest[7][13]
        rf_config = self.config.get('models', {}).get('random_forest', {})
        self.rf_model = RandomForestClassifier(**rf_config)
        self.rf_model.fit(X_train, y_train)
        
        # Train XGBoost[8][14]
        xgb_config = self.config.get('models', {}).get('xgboost', {})
        xgb_config['num_class'] = n_classes
        self.xgb_model = XGBClassifier(**xgb_config)
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate models
        self._evaluate_models(X_val, y_val)
    
    def _evaluate_models(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate trained models on validation data."""
        # Random Forest evaluation
        rf_pred = self.rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_pred)
        logger.info(f"Random Forest validation accuracy: {rf_accuracy:.3f}")
        
        # XGBoost evaluation
        xgb_pred = self.xgb_model.predict(X_val)
        xgb_accuracy = accuracy_score(y_val, xgb_pred)
        logger.info(f"XGBoost validation accuracy: {xgb_accuracy:.3f}")
        
        # Ensemble evaluation
        ensemble_pred = self._ensemble_predict(X_val)
        ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        logger.info(f"Ensemble validation accuracy: {ensemble_accuracy:.3f}")
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions by averaging probabilities."""
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # Average probabilities
        ensemble_proba = (rf_proba + xgb_proba) / 2
        return np.argmax(ensemble_proba, axis=1)
    
    def classify_jailbreak(self, prompt: str) -> Tuple[str, float]:
        """
        Classify jailbreak type with confidence score.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Tuple of (predicted_type, confidence_score)
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Extract features
        X = self.extract_features([prompt])
        
        # Get probabilities from both models
        rf_proba = self.rf_model.predict_proba(X)[0]
        xgb_proba = self.xgb_model.predict_proba(X)[0]
        
        # Average probabilities
        ensemble_proba = (rf_proba + xgb_proba) / 2
        
        # Get prediction and confidence
        predicted_idx = np.argmax(ensemble_proba)
        confidence = float(ensemble_proba[predicted_idx])
        predicted_type = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        logger.debug(f"Predicted type: {predicted_type}, confidence: {confidence:.3f}")
        
        return predicted_type, confidence
    
    def save_models(self, models_dir: str) -> None:
        """Save trained models and transformers."""
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(self.rf_model, models_path / "stage2_rf_model.pkl")
        joblib.dump(self.xgb_model, models_path / "stage2_xgb_model.pkl")
        joblib.dump(self.pca_transformer, models_path / "pca_transformer.pkl")
        joblib.dump(self.label_encoder, models_path / "label_encoder.pkl")
        
        logger.info(f"Models saved to {models_dir}")
    
    def load_models(self, models_dir: str) -> None:
        """Load trained models and transformers."""
        models_path = Path(models_dir)
        
        self.rf_model = joblib.load(models_path / "stage2_rf_model.pkl")
        self.xgb_model = joblib.load(models_path / "stage2_xgb_model.pkl")
        self.pca_transformer = joblib.load(models_path / "pca_transformer.pkl")
        self.label_encoder = joblib.load(models_path / "label_encoder.pkl")
        
        logger.info(f"Models loaded from {models_dir}")
