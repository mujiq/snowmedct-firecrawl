"""
Dependency injection for database managers and other shared resources.
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
import logging

from ..database.postgres_manager import PostgresManager
from ..database.milvus_manager import MilvusManager
from ..embeddings.model_manager import EmbeddingModelManager
from ..graph.janusgraph_manager import JanusGraphManager
from .config import settings

logger = logging.getLogger(__name__)

# Global database manager instances
_database_managers: Optional[Dict[str, Any]] = None
_embedding_model_manager: Optional[EmbeddingModelManager] = None


async def get_database_managers() -> Dict[str, Any]:
    """
    Get database managers singleton.
    
    Returns:
        Dictionary containing database manager instances
    """
    global _database_managers
    
    if _database_managers is None:
        _database_managers = await initialize_database_managers()
    
    return _database_managers


async def initialize_database_managers() -> Dict[str, Any]:
    """
    Initialize all database managers.
    
    Returns:
        Dictionary containing initialized database manager instances
        
    Raises:
        HTTPException: If database initialization fails
    """
    managers = {}
    
    try:
        # Initialize PostgreSQL manager
        try:
            postgres_manager = PostgresManager(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
            postgres_manager.connect()
            managers['postgres'] = postgres_manager
            logger.info("PostgreSQL manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL manager: {e}")
            # Don't fail completely if one database is unavailable
            
        # Initialize Milvus manager
        try:
            milvus_manager = MilvusManager(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD
            )
            milvus_manager.connect()
            managers['milvus'] = milvus_manager
            logger.info("Milvus manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Milvus manager: {e}")
            
        # Initialize JanusGraph manager
        try:
            janusgraph_manager = JanusGraphManager(
                host=settings.JANUSGRAPH_HOST,
                port=settings.JANUSGRAPH_PORT
            )
            janusgraph_manager.connect()
            managers['janusgraph'] = janusgraph_manager
            logger.info("JanusGraph manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize JanusGraph manager: {e}")
            
        if not managers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No database connections could be established"
            )
            
        return managers
        
    except Exception as e:
        logger.error(f"Failed to initialize database managers: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database initialization failed"
        )


async def get_embedding_model_manager() -> EmbeddingModelManager:
    """
    Get embedding model manager singleton.
    
    Returns:
        EmbeddingModelManager instance
    """
    global _embedding_model_manager
    
    if _embedding_model_manager is None:
        _embedding_model_manager = await initialize_embedding_model_manager()
    
    return _embedding_model_manager


async def initialize_embedding_model_manager() -> EmbeddingModelManager:
    """
    Initialize embedding model manager.
    
    Returns:
        Initialized EmbeddingModelManager instance
        
    Raises:
        HTTPException: If model initialization fails
    """
    try:
        model_manager = EmbeddingModelManager(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE
        )
        model_manager.load_model()
        logger.info("Embedding model manager initialized successfully")
        return model_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model initialization failed"
        )


def get_postgres_manager(
    managers: Dict[str, Any] = Depends(get_database_managers)
) -> PostgresManager:
    """
    Get PostgreSQL manager dependency.
    
    Args:
        managers: Database managers dictionary
        
    Returns:
        PostgresManager instance
        
    Raises:
        HTTPException: If PostgreSQL is not available
    """
    if 'postgres' not in managers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL service not available"
        )
    return managers['postgres']


def get_milvus_manager(
    managers: Dict[str, Any] = Depends(get_database_managers)
) -> MilvusManager:
    """
    Get Milvus manager dependency.
    
    Args:
        managers: Database managers dictionary
        
    Returns:
        MilvusManager instance
        
    Raises:
        HTTPException: If Milvus is not available
    """
    if 'milvus' not in managers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Milvus service not available"
        )
    return managers['milvus']


def get_janusgraph_manager(
    managers: Dict[str, Any] = Depends(get_database_managers)
) -> JanusGraphManager:
    """
    Get JanusGraph manager dependency.
    
    Args:
        managers: Database managers dictionary
        
    Returns:
        JanusGraphManager instance
        
    Raises:
        HTTPException: If JanusGraph is not available
    """
    if 'janusgraph' not in managers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JanusGraph service not available"
        )
    return managers['janusgraph']


def get_current_user():
    """
    Get current authenticated user (placeholder for future authentication).
    
    Returns:
        User information (currently returns None for no authentication)
    """
    # TODO: Implement actual authentication
    return None


def check_rate_limit():
    """
    Check rate limiting (placeholder for future implementation).
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # TODO: Implement actual rate limiting
    if settings.RATE_LIMIT_ENABLED:
        # Placeholder - implement Redis-based rate limiting
        pass 