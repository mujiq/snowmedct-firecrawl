"""
Embedding Model Manager for SNOMED-CT Platform

This module handles loading and inference of biomedical embedding models
for generating vector representations of SNOMED-CT concepts.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import warnings

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Suppress some transformers warnings
if DEPENDENCIES_AVAILABLE:
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")


class EmbeddingModelManager:
    """Manages embedding model loading and inference for SNOMED-CT concepts."""
    
    # Recommended biomedical models
    BIOMEDICAL_MODELS = {
        'biobert': 'dmis-lab/biobert-base-cased-v1.1',
        'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'biobert-sentence': 'pritamdeka/S-BioBert-snli-multinli-stsb',
        'clinical-longformer': 'yikuan8/Clinical-Longformer'
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        use_sentence_transformer: bool = False
    ):
        """
        Initialize the embedding model manager.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_length: Maximum sequence length for tokenization
            use_sentence_transformer: Whether to use sentence-transformers library
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies (torch, transformers, etc.) are not installed")
            
        self.model_name = model_name or settings.embedding.model_name
        self.device = self._setup_device(device or settings.embedding.device)
        self.max_length = max_length
        self.use_sentence_transformer = use_sentence_transformer or 'sentence' in self.model_name.lower()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.config = None
        self.embedding_dim = None
        
        logger.info(f"Initialized EmbeddingModelManager with model: {self.model_name}")
        logger.info(f"Device: {self.device}, Max length: {self.max_length}")
    
    def _setup_device(self, device: str):
        """Set up the computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            if self.use_sentence_transformer:
                # Use sentence-transformers for models optimized for sentence embeddings
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer model with embedding dimension: {self.embedding_dim}")
                
            else:
                # Use transformers library for raw BERT models
                self.config = AutoConfig.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name, config=self.config)
                
                # Move model to device
                self.model.to(self.device)
                self.model.eval()
                
                # Get embedding dimension
                self.embedding_dim = self.config.hidden_size
                logger.info(f"Loaded transformer model with embedding dimension: {self.embedding_dim}")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess SNOMED-CT text for embedding generation.
        
        Args:
            text: Raw SNOMED-CT text (e.g., concept term, description)
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove SNOMED-CT identifiers and formatting
        # Pattern to match SNOMED IDs like |12345678|
        text = re.sub(r'\|\d+\|', '', text)
        
        # Remove parenthetical information that might be too specific
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for consistency (except for sentence-transformers which handle this)
        if not self.use_sentence_transformer:
            text = text.lower()
        
        return text
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            if self.use_sentence_transformer:
                # Use sentence-transformers
                embedding = self.model.encode(processed_text, convert_to_numpy=True)
                return embedding.astype(np.float32)
            
            else:
                # Use raw transformers model
                if not self.tokenizer:
                    raise ValueError("Tokenizer not loaded. Call load_model() first.")
                    
                with torch.no_grad():
                    # Tokenize
                    inputs = self.tokenizer(
                        processed_text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model outputs
                    outputs = self.model(**inputs)
                    
                    # Use mean pooling of last hidden states
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Apply attention mask and compute mean
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings_masked = embeddings * mask_expanded
                    sum_embeddings = torch.sum(embeddings_masked, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embedding = sum_embeddings / sum_mask
                    
                    return embedding.cpu().numpy().astype(np.float32).flatten()
        
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:100]}... Error: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses settings if not provided)
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        batch_size = batch_size or settings.embedding.batch_size
        embeddings: List[np.ndarray] = []
        
        # Process in batches
        total_texts = len(texts)
        batches = [texts[i:i + batch_size] for i in range(0, total_texts, batch_size)]
        
        progress_bar = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for batch in progress_bar:
            if self.use_sentence_transformer:
                # Process batch with sentence-transformers
                processed_batch = [self.preprocess_text(text) for text in batch]
                batch_embeddings = self.model.encode(
                    processed_batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])
                
            else:
                # Process batch with transformers
                for text in batch:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
        
        embeddings_count = len(embeddings)
        logger.info(f"Generated {embeddings_count} embeddings")
        return embeddings
    
    def test_embedding_generation(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test embedding generation with sample SNOMED-CT texts.
        
        Args:
            sample_texts: Optional list of sample texts to test
            
        Returns:
            Test results dictionary
        """
        # Default sample SNOMED-CT terms
        if sample_texts is None:
            sample_texts = [
                "Myocardial infarction",
                "Type 2 diabetes mellitus",
                "Hypertensive disorder",
                "Pneumonia",
                "Fracture of bone",
                "Malignant neoplasm of breast",
                "Chronic kidney disease",
                "Asthma",
                "Depression",
                "Osteoarthritis"
            ]
        
        logger.info("Testing embedding generation with sample texts")
        
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(sample_texts, show_progress=True)
            
            # Calculate statistics
            embedding_matrix = np.array(embeddings)
            
            results = {
                'success': True,
                'num_samples': len(sample_texts),
                'embedding_dimension': self.embedding_dim,
                'embedding_stats': {
                    'mean': float(np.mean(embedding_matrix)),
                    'std': float(np.std(embedding_matrix)),
                    'min': float(np.min(embedding_matrix)),
                    'max': float(np.max(embedding_matrix))
                },
                'sample_texts': sample_texts[:5],  # First 5 samples
                'model_info': {
                    'model_name': self.model_name,
                    'device': str(self.device),
                    'use_sentence_transformer': self.use_sentence_transformer
                }
            }
            
            # Test similarity between related concepts
            embeddings_count = len(embeddings)
            if embeddings_count >= 2:
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                results['sample_similarity'] = float(similarity)
            
            logger.info("Embedding generation test completed successfully")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info(f"Sample embedding stats: mean={results['embedding_stats']['mean']:.4f}, "
                       f"std={results['embedding_stats']['std']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding generation test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_info': {
                    'model_name': self.model_name,
                    'device': str(self.device)
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': str(self.device),
            'max_length': self.max_length,
            'use_sentence_transformer': self.use_sentence_transformer,
            'model_loaded': self.model is not None
        }


def get_recommended_model(use_case: str = 'general') -> str:
    """
    Get recommended model for specific use case.
    
    Args:
        use_case: Use case ('general', 'clinical', 'fast', 'accurate')
        
    Returns:
        Model name
    """
    recommendations = {
        'general': EmbeddingModelManager.BIOMEDICAL_MODELS['biobert'],
        'clinical': EmbeddingModelManager.BIOMEDICAL_MODELS['clinicalbert'],
        'fast': EmbeddingModelManager.BIOMEDICAL_MODELS['biobert-sentence'],
        'accurate': EmbeddingModelManager.BIOMEDICAL_MODELS['pubmedbert']
    }
    
    return recommendations.get(use_case, EmbeddingModelManager.BIOMEDICAL_MODELS['biobert']) 