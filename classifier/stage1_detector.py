"""
Stage 1: Fast binary jailbreak detection using embeddings and regex patterns.
"""

import faiss
import numpy as np
import regex as re
import logging
import pickle
from typing import List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Stage1Detector:
    """Fast binary jailbreak detector using semantic similarity and regex patterns."""
    
    def __init__(self, config: dict):
        """
        Initialize Stage 1 detector.
        
        Args:
            config: Configuration dictionary containing model and threshold settings
        """
        self.config = config
        self.embedding_model = None
        self.faiss_index = None
        self.regex_patterns = []
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.k_neighbors = config.get('k_neighbors', 5)
        
        # Compile regex patterns[9]
        for pattern in config.get('regex_patterns', []):
            try:
                self.regex_patterns.append(re.compile(pattern))
                logger.debug(f"Compiled regex pattern: {pattern}")
            except re.error as e:
                logger.warning(f"Failed to compile regex pattern '{pattern}': {e}")
    
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
    
    def build_faiss_index(self, unsafe_prompts: List[str]) -> None:
        """
        Build FAISS index from unsafe prompts.
        
        Args:
            unsafe_prompts: List of unsafe prompt strings
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        logger.info(f"Building FAISS index from {len(unsafe_prompts)} unsafe prompts")
        
        # Generate embeddings for unsafe prompts
        embeddings = self.embedding_model.encode(unsafe_prompts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity[6][12]
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings)
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors of dimension {dimension}")
    
    def save_index(self, index_path: str) -> None:
        """Save FAISS index to disk."""
        if self.faiss_index is None:
            raise ValueError("No FAISS index to save. Build index first.")
        
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """Load FAISS index from disk."""
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.faiss_index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded from {index_path}")
    
    def check_regex_patterns(self, prompt: str) -> bool:
        """
        Check if prompt matches any regex patterns.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            True if any pattern matches, False otherwise
        """
        for pattern in self.regex_patterns:
            if pattern.search(prompt):
                logger.debug(f"Regex match found: {pattern.pattern}")
                return True
        return False
    
    def compute_similarity(self, prompt: str) -> float:
        """
        Compute maximum cosine similarity with unsafe prompts in FAISS index.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Maximum cosine similarity score
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        if self.faiss_index is None:
            logger.warning("No FAISS index loaded, returning 0.0 similarity")
            return 0.0
        
        # Generate embedding for input prompt
        embedding = self.embedding_model.encode([prompt])
        embedding = np.array(embedding).astype('float32')
        faiss.normalize_L2(embedding)
        
        # Search for k nearest neighbors
        similarities, _ = self.faiss_index.search(embedding, self.k_neighbors)
        
        # Return maximum similarity
        max_similarity = float(similarities[0].max()) if len(similarities[0]) > 0 else 0.0
        logger.debug(f"Max similarity: {max_similarity}")
        
        return max_similarity
    
    def detect_jailbreak(self, prompt: str) -> bool:
        """
        Detect if prompt is a jailbreak attempt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            True if jailbreak detected, False otherwise
        """
        # Check regex patterns first (faster)
        if self.check_regex_patterns(prompt):
            logger.debug("Jailbreak detected via regex patterns")
            return True
        
        # Check semantic similarity
        similarity = self.compute_similarity(prompt)
        if similarity > self.similarity_threshold:
            logger.debug(f"Jailbreak detected via similarity: {similarity} > {self.similarity_threshold}")
            return True
        
        logger.debug("No jailbreak detected")
        return False
