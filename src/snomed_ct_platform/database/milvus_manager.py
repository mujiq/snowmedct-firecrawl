"""
Milvus Database Manager for SNOMED-CT Platform

This module handles Milvus vector database operations including collection setup,
embedding storage, and similarity search for SNOMED-CT concepts.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import time

try:
    import numpy as np
    from pymilvus import (
        connections, utility, Collection, CollectionSchema, FieldSchema, 
        DataType, db, IndexType, MetricType
    )
    MILVUS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Milvus dependencies not available: {e}")
    MILVUS_AVAILABLE = False

from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MilvusManager:
    """Manages Milvus vector database operations for SNOMED-CT embeddings."""
    
    # Available index types for different use cases
    INDEX_TYPES = {
        'hnsw': IndexType.HNSW,           # High performance, memory intensive
        'ivf_flat': IndexType.IVF_FLAT,   # Good balance
        'ivf_sq8': IndexType.IVF_SQ8,     # Memory efficient
        'ivf_pq': IndexType.IVF_PQ,       # Highly compressed
        'annoy': IndexType.ANNOY,         # Fast build time
    }
    
    # Available metric types
    METRIC_TYPES = {
        'cosine': MetricType.COSINE,      # Cosine similarity (recommended for embeddings)
        'ip': MetricType.IP,              # Inner product
        'l2': MetricType.L2,              # Euclidean distance
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize Milvus manager.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus dependencies not available. Install with: pip install pymilvus")
        
        self.host = host or settings.database.milvus_host
        self.port = port or settings.database.milvus_port
        self.collection_name = collection_name or settings.database.milvus_collection_name
        self.embedding_dim = embedding_dim or settings.embedding.dimension
        
        self.collection: Optional[Collection] = None
        self.connected = False
        
        logger.info(f"Initialized MilvusManager for {self.host}:{self.port}")
        logger.info(f"Collection: {self.collection_name}, Embedding dim: {self.embedding_dim}")
    
    def connect(self) -> None:
        """Connect to Milvus server."""
        try:
            logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # Test connection
            if utility.get_server_version():
                self.connected = True
                logger.info("Successfully connected to Milvus")
                logger.info(f"Milvus server version: {utility.get_server_version()}")
            else:
                raise ConnectionError("Failed to get server version")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection_schema(self) -> CollectionSchema:
        """
        Create collection schema for SNOMED-CT embeddings.
        
        Returns:
            Collection schema
        """
        logger.info("Creating collection schema for SNOMED-CT embeddings")
        
        # Define fields
        fields = [
            # Primary key: SNOMED-CT concept ID
            FieldSchema(
                name="concept_id",
                dtype=DataType.INT64,
                is_primary=True,
                description="SNOMED-CT concept identifier"
            ),
            
            # Embedding vector
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Concept embedding vector"
            ),
            
            # Concept term for reference
            FieldSchema(
                name="term",
                dtype=DataType.VARCHAR,
                max_length=1024,
                description="Fully specified name of the concept"
            ),
            
            # Active status
            FieldSchema(
                name="active",
                dtype=DataType.BOOL,
                description="Whether the concept is active"
            ),
            
            # Module ID for categorization
            FieldSchema(
                name="module_id",
                dtype=DataType.INT64,
                description="SNOMED-CT module identifier"
            ),
            
            # Timestamp for tracking
            FieldSchema(
                name="created_time",
                dtype=DataType.INT64,
                description="Creation timestamp"
            )
        ]
        
        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="SNOMED-CT concept embeddings collection",
            enable_dynamic_field=True
        )
        
        logger.info(f"Created schema with {len(fields)} fields")
        return schema
    
    def create_collection(self, drop_existing: bool = False) -> None:
        """
        Create Milvus collection for SNOMED-CT embeddings.
        
        Args:
            drop_existing: Whether to drop existing collection
        """
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    logger.warning(f"Dropping existing collection: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    self.collection = Collection(self.collection_name)
                    return
            
            # Create new collection
            logger.info(f"Creating new collection: {self.collection_name}")
            schema = self.create_collection_schema()
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def create_index(
        self,
        index_type: str = 'hnsw',
        metric_type: str = 'cosine',
        index_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create index on the embedding field.
        
        Args:
            index_type: Type of index to create
            metric_type: Distance metric to use
            index_params: Additional index parameters
        """
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        try:
            logger.info(f"Creating {index_type} index with {metric_type} metric")
            
            # Default index parameters
            default_params = {
                'hnsw': {"M": 16, "efConstruction": 256},
                'ivf_flat': {"nlist": 1024},
                'ivf_sq8': {"nlist": 1024},
                'ivf_pq': {"nlist": 1024, "m": 16, "nbits": 8},
                'annoy': {"n_trees": 8}
            }
            
            if index_params is None:
                index_params = default_params.get(index_type, {})
            
            # Create index
            index_params_full = {
                "index_type": self.INDEX_TYPES[index_type],
                "metric_type": self.METRIC_TYPES[metric_type],
                "params": index_params
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params_full
            )
            
            logger.info("Index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        try:
            logger.info("Loading collection into memory")
            self.collection.load()
            logger.info("Collection loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
    
    def insert_embeddings(
        self,
        concept_ids: List[int],
        embeddings: List[np.ndarray],
        terms: List[str],
        active_flags: List[bool],
        module_ids: List[int]
    ) -> List[int]:
        """
        Insert embeddings into the collection.
        
        Args:
            concept_ids: List of concept IDs
            embeddings: List of embedding vectors
            terms: List of concept terms
            active_flags: List of active status flags
            module_ids: List of module IDs
            
        Returns:
            List of inserted IDs
        """
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        if not all(len(lst) == len(concept_ids) for lst in [embeddings, terms, active_flags, module_ids]):
            raise ValueError("All input lists must have the same length")
        
        try:
            current_time = int(time.time())
            
            # Prepare data
            data = [
                concept_ids,
                [emb.tolist() if isinstance(emb, np.ndarray) else list(emb) for emb in embeddings],
                terms,
                active_flags,
                module_ids,
                [current_time] * len(concept_ids)
            ]
            
            # Insert data
            logger.info(f"Inserting {len(concept_ids)} embeddings")
            result = self.collection.insert(data)
            
            # Flush to ensure data is written
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(concept_ids)} embeddings")
            return result.primary_keys
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    def search_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar concepts.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            search_params: Search parameters
            output_fields: Fields to include in results
            
        Returns:
            List of search results
        """
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        try:
            # Default search parameters
            if search_params is None:
                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
            
            # Default output fields
            if output_fields is None:
                output_fields = ["concept_id", "term", "active", "module_id"]
            
            # Prepare query vector
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)  # Ensure it's a list
            
            # Perform search
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        'concept_id': hit.id,
                        'distance': hit.distance,
                        'score': 1 - hit.distance if hit.distance <= 1 else 0  # Convert distance to similarity score
                    }
                    # Add entity fields
                    if hasattr(hit, 'entity'):
                        for field in output_fields:
                            if hasattr(hit.entity, field):
                                result[field] = getattr(hit.entity, field)
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar concepts")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar concepts: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        try:
            stats = {
                'name': self.collection.name,
                'description': self.collection.description,
                'num_entities': self.collection.num_entities,
                'schema': {
                    'fields': [
                        {
                            'name': field.name,
                            'type': str(field.dtype),
                            'description': field.description
                        }
                        for field in self.collection.schema.fields
                    ]
                }
            }
            
            # Get index information
            try:
                indexes = self.collection.indexes
                stats['indexes'] = [
                    {
                        'field_name': idx.field_name,
                        'index_type': str(idx.params.get('index_type', 'Unknown')),
                        'metric_type': str(idx.params.get('metric_type', 'Unknown'))
                    }
                    for idx in indexes
                ]
            except:
                stats['indexes'] = []
            
            logger.info(f"Collection stats: {stats['num_entities']} entities")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    def delete_by_ids(self, concept_ids: List[int]) -> int:
        """
        Delete concepts by IDs.
        
        Args:
            concept_ids: List of concept IDs to delete
            
        Returns:
            Number of deleted entities
        """
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection() first.")
        
        try:
            logger.info(f"Deleting {len(concept_ids)} concepts")
            
            # Create expression for deletion
            ids_str = ','.join(map(str, concept_ids))
            expr = f"concept_id in [{ids_str}]"
            
            # Delete entities
            result = self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Successfully deleted {len(concept_ids)} concepts")
            return len(concept_ids)
            
        except Exception as e:
            logger.error(f"Failed to delete concepts: {e}")
            raise
    
    def test_operations(self) -> Dict[str, Any]:
        """
        Test basic Milvus operations.
        
        Returns:
            Test results
        """
        logger.info("Testing Milvus operations")
        
        try:
            # Test data
            test_concept_ids = [1000001, 1000002, 1000003]
            test_embeddings = [
                np.random.rand(self.embedding_dim).astype(np.float32),
                np.random.rand(self.embedding_dim).astype(np.float32),
                np.random.rand(self.embedding_dim).astype(np.float32)
            ]
            test_terms = ["Test concept 1", "Test concept 2", "Test concept 3"]
            test_active = [True, True, False]
            test_modules = [900000000000207008, 900000000000207008, 900000000000207008]
            
            # Insert test data
            inserted_ids = self.insert_embeddings(
                test_concept_ids, test_embeddings, test_terms, test_active, test_modules
            )
            
            # Test search
            search_results = self.search_similar(test_embeddings[0], limit=5)
            
            # Clean up test data
            self.delete_by_ids(test_concept_ids)
            
            results = {
                'success': True,
                'operations_tested': ['insert', 'search', 'delete'],
                'test_data_size': len(test_concept_ids),
                'inserted_ids_count': len(inserted_ids),
                'search_results_count': len(search_results),
                'collection_stats': self.get_collection_stats()
            }
            
            logger.info("Milvus operations test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Milvus operations test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            } 