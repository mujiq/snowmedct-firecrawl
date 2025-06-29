"""
API Configuration settings for SNOMED-CT Multi-Modal Data Platform.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API configuration settings."""
    
    # API Server settings
    HOST: str = Field(default="localhost", description="API server host")
    PORT: int = Field(default=8000, description="API server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # Database settings
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="snomed_ct", description="PostgreSQL database")
    
    MILVUS_HOST: str = Field(default="localhost", description="Milvus host")
    MILVUS_PORT: int = Field(default=19530, description="Milvus port")
    MILVUS_USER: Optional[str] = Field(default=None, description="Milvus user (optional)")
    MILVUS_PASSWORD: Optional[str] = Field(default=None, description="Milvus password (optional)")
    
    JANUSGRAPH_HOST: str = Field(default="localhost", description="JanusGraph host")
    JANUSGRAPH_PORT: int = Field(default=8182, description="JanusGraph port")
    
    # API settings
    API_V1_STR: str = Field(default="/api/v1", description="API version prefix")
    PROJECT_NAME: str = Field(default="SNOMED-CT Multi-Modal Platform", description="Project name")
    
    # Pagination settings
    DEFAULT_PAGE_SIZE: int = Field(default=20, description="Default pagination page size")
    MAX_PAGE_SIZE: int = Field(default=1000, description="Maximum pagination page size")
    
    # Search settings
    DEFAULT_SEARCH_LIMIT: int = Field(default=10, description="Default search result limit")
    MAX_SEARCH_LIMIT: int = Field(default=100, description="Maximum search result limit")
    
    # Embedding settings
    EMBEDDING_MODEL: str = Field(default="clinicalbert", description="Default embedding model")
    EMBEDDING_DIMENSION: int = Field(default=768, description="Embedding vector dimension")
    
    # Cache settings
    ENABLE_CACHE: bool = Field(default=True, description="Enable response caching")
    CACHE_TTL: int = Field(default=300, description="Cache TTL in seconds")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per minute")
    
    # Security settings
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", description="Secret key for JWT")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings() 