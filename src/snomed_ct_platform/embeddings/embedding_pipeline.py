"""
Embedding Generation Pipeline for SNOMED-CT Platform

This module orchestrates the complete embedding generation and storage process,
integrating the embedding model manager with Milvus vector database.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .model_manager import EmbeddingModelManager
from ..database.milvus_manager import MilvusManager
from ..database.postgres_manager import PostgresManager
from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for embedding generation pipeline."""
    start_time: float
    end_time: Optional[float] = None
    concepts_processed: int = 0
    embeddings_generated: int = 0
    embeddings_stored: int = 0
    errors: int = 0
    batch_count: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get pipeline duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def embeddings_per_second(self) -> Optional[float]:
        """Get embeddings generated per second."""
        if self.duration and self.duration > 0:
            return self.embeddings_generated / self.duration
        return None


class EmbeddingPipeline:
    """Pipeline for generating and storing SNOMED-CT concept embeddings."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_batch_size: Optional[int] = None,
        storage_batch_size: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: Embedding model to use
            embedding_batch_size: Batch size for embedding generation
            storage_batch_size: Batch size for storage operations
            device: Device to run the model on
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for embedding pipeline")
        
        self.embedding_batch_size = embedding_batch_size or settings.embedding.batch_size
        self.storage_batch_size = storage_batch_size or settings.application.batch_size
        
        # Initialize components
        self.model_manager = EmbeddingModelManager(
            model_name=model_name,
            device=device
        )
        self.milvus_manager = MilvusManager()
        self.postgres_manager = PostgresManager()
        
        # Pipeline state
        self.model_loaded = False
        self.milvus_connected = False
        self.postgres_connected = False
        
        # Statistics
        self.stats = EmbeddingStats(start_time=time.time())
        
        logger.info("Initialized EmbeddingPipeline")
        logger.info(f"Embedding batch size: {self.embedding_batch_size}")
        logger.info(f"Storage batch size: {self.storage_batch_size}")
    
    def setup_components(
        self,
        load_model: bool = True,
        setup_milvus: bool = True,
        setup_postgres: bool = True,
        create_collection: bool = True,
        create_index: bool = True
    ) -> None:
        """
        Set up all pipeline components.
        
        Args:
            load_model: Whether to load the embedding model
            setup_milvus: Whether to set up Milvus connection
            setup_postgres: Whether to set up PostgreSQL connection
            create_collection: Whether to create Milvus collection
            create_index: Whether to create Milvus index
        """
        logger.info("Setting up embedding pipeline components")
        
        try:
            # Load embedding model
            if load_model:
                logger.info("Loading embedding model...")
                self.model_manager.load_model()
                self.model_loaded = True
                logger.info(f"Model loaded: {self.model_manager.model_name}")
                logger.info(f"Embedding dimension: {self.model_manager.embedding_dim}")
            
            # Set up Milvus
            if setup_milvus:
                logger.info("Setting up Milvus connection...")
                
                # Update embedding dimension in Milvus manager
                if self.model_loaded and self.model_manager.embedding_dim:
                    self.milvus_manager.embedding_dim = self.model_manager.embedding_dim
                
                self.milvus_manager.connect()
                self.milvus_connected = True
                
                if create_collection:
                    self.milvus_manager.create_collection(drop_existing=False)
                    
                    if create_index:
                        self.milvus_manager.create_index(
                            index_type='hnsw',
                            metric_type='cosine'
                        )
                    
                    self.milvus_manager.load_collection()
                
                logger.info("Milvus setup completed")
            
            # Set up PostgreSQL
            if setup_postgres:
                logger.info("Setting up PostgreSQL connection...")
                self.postgres_manager.connect()
                self.postgres_connected = True
                logger.info("PostgreSQL setup completed")
            
            logger.info("All components set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up components: {e}")
            raise
    
    def get_concepts_for_embedding(
        self,
        limit: Optional[int] = None,
        only_active: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get concepts from PostgreSQL that need embeddings.
        
        Args:
            limit: Maximum number of concepts to retrieve
            only_active: Whether to only get active concepts
            
        Returns:
            List of concept dictionaries
        """
        if not self.postgres_connected:
            raise ValueError("PostgreSQL not connected. Call setup_components() first.")
        
        try:
            logger.info("Retrieving concepts for embedding generation")
            concepts = self.postgres_manager.get_active_concepts(limit=limit)
            
            logger.info(f"Retrieved {len(concepts)} concepts for embedding")
            return concepts
            
        except Exception as e:
            logger.error(f"Failed to get concepts: {e}")
            raise
    
    def generate_embeddings_for_concepts(
        self,
        concepts: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Tuple[List[int], List[np.ndarray], List[str], List[bool], List[int]]:
        """
        Generate embeddings for a list of concepts.
        
        Args:
            concepts: List of concept dictionaries
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (concept_ids, embeddings, terms, active_flags, module_ids)
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call setup_components() first.")
        
        try:
            logger.info(f"Generating embeddings for {len(concepts)} concepts")
            
            # Extract data
            concept_ids = [concept['id'] for concept in concepts]
            terms = [concept.get('fully_specified_name', '') or '' for concept in concepts]
            active_flags = [concept.get('active', True) for concept in concepts]
            module_ids = [concept.get('module_id', 0) for concept in concepts]
            
            # Generate embeddings
            embeddings = self.model_manager.generate_embeddings_batch(
                terms,
                batch_size=self.embedding_batch_size,
                show_progress=show_progress
            )
            
            # Update statistics
            self.stats.concepts_processed += len(concepts)
            self.stats.embeddings_generated += len(embeddings)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return concept_ids, embeddings, terms, active_flags, module_ids
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self.stats.errors += 1
            raise
    
    def store_embeddings_batch(
        self,
        concept_ids: List[int],
        embeddings: List[np.ndarray],
        terms: List[str],
        active_flags: List[bool],
        module_ids: List[int]
    ) -> int:
        """
        Store embeddings in Milvus in batches.
        
        Args:
            concept_ids: List of concept IDs
            embeddings: List of embedding vectors
            terms: List of concept terms
            active_flags: List of active flags
            module_ids: List of module IDs
            
        Returns:
            Number of embeddings stored
        """
        if not self.milvus_connected:
            raise ValueError("Milvus not connected. Call setup_components() first.")
        
        try:
            stored_count = 0
            total_items = len(concept_ids)
            
            # Process in storage batches
            for i in range(0, total_items, self.storage_batch_size):
                end_idx = min(i + self.storage_batch_size, total_items)
                
                batch_concept_ids = concept_ids[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_terms = terms[i:end_idx]
                batch_active = active_flags[i:end_idx]
                batch_modules = module_ids[i:end_idx]
                
                # Store batch
                inserted_ids = self.milvus_manager.insert_embeddings(
                    batch_concept_ids,
                    batch_embeddings,
                    batch_terms,
                    batch_active,
                    batch_modules
                )
                
                stored_count += len(inserted_ids)
                self.stats.batch_count += 1
                
                logger.debug(f"Stored batch {self.stats.batch_count}: {len(inserted_ids)} embeddings")
            
            self.stats.embeddings_stored += stored_count
            logger.info(f"Stored {stored_count} embeddings in {self.stats.batch_count} batches")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            self.stats.errors += 1
            raise
    
    def run_embedding_pipeline(
        self,
        concept_limit: Optional[int] = None,
        test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete embedding generation and storage pipeline.
        
        Args:
            concept_limit: Maximum number of concepts to process
            test_mode: Whether to run in test mode (small sample)
            
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting embedding generation pipeline")
        
        if test_mode:
            concept_limit = min(concept_limit or 100, 100)
            logger.info(f"Running in test mode with limit: {concept_limit}")
        
        try:
            # Set up components
            self.setup_components()
            
            # Get concepts to process
            concepts = self.get_concepts_for_embedding(limit=concept_limit)
            
            if not concepts:
                logger.warning("No concepts found for embedding generation")
                return {
                    'success': True,
                    'message': 'No concepts to process',
                    'stats': self._get_pipeline_stats()
                }
            
            # Generate embeddings
            concept_ids, embeddings, terms, active_flags, module_ids = \
                self.generate_embeddings_for_concepts(concepts)
            
            # Store embeddings
            stored_count = self.store_embeddings_batch(
                concept_ids, embeddings, terms, active_flags, module_ids
            )
            
            # Update final statistics
            self.stats.end_time = time.time()
            
            # Get collection statistics
            collection_stats = self.milvus_manager.get_collection_stats()
            
            # Create summary
            summary = {
                'success': True,
                'pipeline_stats': self._get_pipeline_stats(),
                'collection_stats': collection_stats,
                'model_info': self.model_manager.get_model_info()
            }
            
            logger.info(f"Embedding pipeline completed successfully")
            logger.info(f"Processed: {self.stats.concepts_processed} concepts")
            logger.info(f"Generated: {self.stats.embeddings_generated} embeddings")
            logger.info(f"Stored: {self.stats.embeddings_stored} embeddings")
            logger.info(f"Duration: {self.stats.duration:.2f} seconds")
            logger.info(f"Speed: {self.stats.embeddings_per_second:.2f} embeddings/sec")
            
            return summary
            
        except Exception as e:
            self.stats.end_time = time.time()
            logger.error(f"Embedding pipeline failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_stats': self._get_pipeline_stats()
            }
    
    def test_pipeline(self) -> Dict[str, Any]:
        """
        Test the embedding pipeline with sample data.
        
        Returns:
            Test results
        """
        logger.info("Testing embedding pipeline")
        
        try:
            # Set up components
            self.setup_components(create_collection=True, create_index=True)
            
            # Test embedding generation
            model_test = self.model_manager.test_embedding_generation()
            
            # Test Milvus operations
            milvus_test = self.milvus_manager.test_operations()
            
            # Run mini pipeline
            pipeline_test = self.run_embedding_pipeline(
                concept_limit=10,
                test_mode=True
            )
            
            results = {
                'success': True,
                'model_test': model_test,
                'milvus_test': milvus_test,
                'pipeline_test': pipeline_test,
                'components_status': {
                    'model_loaded': self.model_loaded,
                    'milvus_connected': self.milvus_connected,
                    'postgres_connected': self.postgres_connected
                }
            }
            
            logger.info("Embedding pipeline test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Embedding pipeline test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'duration_seconds': self.stats.duration,
            'concepts_processed': self.stats.concepts_processed,
            'embeddings_generated': self.stats.embeddings_generated,
            'embeddings_stored': self.stats.embeddings_stored,
            'batch_count': self.stats.batch_count,
            'errors': self.stats.errors,
            'embeddings_per_second': self.stats.embeddings_per_second
        }


def main():
    """Main entry point for the embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SNOMED-CT Embedding Generation Pipeline")
    parser.add_argument("--model", help="Embedding model to use")
    parser.add_argument("--limit", type=int, help="Maximum number of concepts to process")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--storage-batch-size", type=int, default=1000, help="Storage batch size")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(
        model_name=args.model,
        embedding_batch_size=args.embedding_batch_size,
        storage_batch_size=args.storage_batch_size,
        device=args.device
    )
    
    # Run pipeline
    if args.test:
        summary = pipeline.test_pipeline()
    else:
        summary = pipeline.run_embedding_pipeline(concept_limit=args.limit)
    
    # Print summary
    if summary['success']:
        print("\n✅ Embedding pipeline completed successfully!")
        if 'pipeline_stats' in summary:
            stats = summary['pipeline_stats']
            print(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")
            print(f"Embeddings generated: {stats.get('embeddings_generated', 0)}")
            print(f"Embeddings stored: {stats.get('embeddings_stored', 0)}")
            print(f"Speed: {stats.get('embeddings_per_second', 0):.2f} embeddings/sec")
    else:
        print(f"\n❌ Embedding pipeline failed: {summary.get('error', 'Unknown error')}")
    
    return 0 if summary['success'] else 1


if __name__ == "__main__":
    exit(main()) 