"""
Configuration Settings for SNOMED-CT Platform

This module manages all configuration settings using environment variables
with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "snomed_ct"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "snomed_ct_embeddings"
    
    # JanusGraph
    janusgraph_host: str = "localhost"
    janusgraph_port: int = 8182
    janusgraph_graph_name: str = "snomed_ct"
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def janusgraph_url(self) -> str:
        """Get JanusGraph WebSocket URL."""
        return f"ws://{self.janusgraph_host}:{self.janusgraph_port}/gremlin"


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration settings."""
    
    model_name: str = Field(default="dmis-lab/biobert-base-cased-v1.1", env="EMBEDDING_MODEL")
    dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    device: str = Field(default="cpu", env="DEVICE")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")


class ApplicationSettings(BaseSettings):
    """General application configuration settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    
    class Config:
        env_file = ".env"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    debug: bool = Field(default=False, env="API_DEBUG")


class Settings:
    """Main settings container."""
    
    def __init__(self):
        self.database = DatabaseSettings()
        self.embedding = EmbeddingSettings()
        self.application = ApplicationSettings()
        self.api = APISettings()


# Global settings instance
settings = Settings() 