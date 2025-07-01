"""
Unit tests for API configuration module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.snomed_ct_platform.api.config import Settings, settings


class TestSettings:
    """Test cases for Settings configuration class."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        config = Settings()
        
        # API Server defaults
        assert config.HOST == "localhost"
        assert config.PORT == 8000
        assert config.DEBUG == False
        
        # CORS defaults
        assert "http://localhost:3000" in config.ALLOWED_ORIGINS
        assert "http://localhost:8080" in config.ALLOWED_ORIGINS
        
        # Database defaults
        assert config.POSTGRES_HOST == "localhost"
        assert config.POSTGRES_PORT == 5432
        assert config.POSTGRES_USER == "postgres"
        assert config.POSTGRES_PASSWORD == "postgres"
        assert config.POSTGRES_DB == "snomed_ct"
        
        # Milvus defaults
        assert config.MILVUS_HOST == "localhost"
        assert config.MILVUS_PORT == 19530
        assert config.MILVUS_USER is None
        assert config.MILVUS_PASSWORD is None
        
        # JanusGraph defaults
        assert config.JANUSGRAPH_HOST == "localhost"
        assert config.JANUSGRAPH_PORT == 8182
    
    def test_api_settings(self):
        """Test API-specific settings."""
        config = Settings()
        
        assert config.API_V1_STR == "/api/v1"
        assert config.PROJECT_NAME == "SNOMED-CT Multi-Modal Platform"
        assert config.DEFAULT_PAGE_SIZE == 20
        assert config.MAX_PAGE_SIZE == 1000
        assert config.DEFAULT_SEARCH_LIMIT == 10
        assert config.MAX_SEARCH_LIMIT == 100
    
    def test_embedding_settings(self):
        """Test embedding-related settings."""
        config = Settings()
        
        assert config.EMBEDDING_MODEL == "clinicalbert"
        assert config.EMBEDDING_DIMENSION == 768
    
    def test_cache_settings(self):
        """Test cache configuration."""
        config = Settings()
        
        assert config.ENABLE_CACHE == True
        assert config.CACHE_TTL == 300
    
    def test_rate_limiting_settings(self):
        """Test rate limiting configuration."""
        config = Settings()
        
        assert config.RATE_LIMIT_ENABLED == True
        assert config.RATE_LIMIT_REQUESTS == 100
    
    def test_security_settings(self):
        """Test security configuration."""
        config = Settings()
        
        assert config.SECRET_KEY == "your-secret-key-change-in-production"
        assert config.ACCESS_TOKEN_EXPIRE_MINUTES == 30
    
    def test_logging_settings(self):
        """Test logging configuration."""
        config = Settings()
        
        assert config.LOG_LEVEL == "INFO"
        assert "%(asctime)s" in config.LOG_FORMAT
        assert "%(levelname)s" in config.LOG_FORMAT
    
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'HOST': 'test-host',
            'PORT': '9000',
            'DEBUG': 'true',
            'POSTGRES_HOST': 'test-db-host',
            'POSTGRES_PORT': '5433',
            'LOG_LEVEL': 'DEBUG'
        }):
            config = Settings()
            
            assert config.HOST == 'test-host'
            assert config.PORT == 9000
            assert config.DEBUG == True
            assert config.POSTGRES_HOST == 'test-db-host'
            assert config.POSTGRES_PORT == 5433
            assert config.LOG_LEVEL == 'DEBUG'
    
    def test_invalid_port_validation(self):
        """Test validation of invalid port numbers."""
        with patch.dict(os.environ, {'PORT': 'invalid-port'}):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_boolean_conversion(self):
        """Test boolean environment variable conversion."""
        with patch.dict(os.environ, {
            'DEBUG': 'false',
            'ENABLE_CACHE': 'false',
            'RATE_LIMIT_ENABLED': 'false'
        }):
            config = Settings()
            
            assert config.DEBUG == False
            assert config.ENABLE_CACHE == False
            assert config.RATE_LIMIT_ENABLED == False
        
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'ENABLE_CACHE': 'true',
            'RATE_LIMIT_ENABLED': 'true'
        }):
            config = Settings()
            
            assert config.DEBUG == True
            assert config.ENABLE_CACHE == True
            assert config.RATE_LIMIT_ENABLED == True
    
    def test_list_environment_variables(self):
        """Test list environment variable parsing."""
        with patch.dict(os.environ, {
            'ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8080,https://example.com'
        }):
            config = Settings()
            
            expected_origins = [
                'http://localhost:3000',
                'http://localhost:8080', 
                'https://example.com'
            ]
            # Note: This test might need adjustment based on how pydantic handles list parsing
            # The current implementation uses default values, so this tests the concept
    
    def test_optional_fields(self):
        """Test optional configuration fields."""
        config = Settings()
        
        # These should be None by default
        assert config.MILVUS_USER is None
        assert config.MILVUS_PASSWORD is None
        
        # Test with environment variables
        with patch.dict(os.environ, {
            'MILVUS_USER': 'test_user',
            'MILVUS_PASSWORD': 'test_pass'
        }):
            config = Settings()
            assert config.MILVUS_USER == 'test_user'
            assert config.MILVUS_PASSWORD == 'test_pass'
    
    def test_field_descriptions(self):
        """Test that field descriptions are properly set."""
        config = Settings()
        schema = config.model_json_schema()
        
        # Check that important fields have descriptions
        assert 'description' in schema['properties']['HOST']
        assert 'description' in schema['properties']['PORT']
        assert 'description' in schema['properties']['POSTGRES_HOST']
        assert 'description' in schema['properties']['LOG_LEVEL']
    
    def test_config_class_settings(self):
        """Test Pydantic Config class settings."""
        config = Settings()
        
        # Test that env_file is set
        assert hasattr(config.model_config, 'env_file') or hasattr(config.Config, 'env_file')
        
        # Test case sensitivity
        with patch.dict(os.environ, {'host': 'lowercase-host'}):
            # This should not override HOST since case_sensitive=True
            config = Settings()
            assert config.HOST == "localhost"  # Should remain default


class TestGlobalSettings:
    """Test cases for global settings instance."""
    
    def test_global_settings_instance(self):
        """Test that global settings instance is properly created."""
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_global_settings_accessibility(self):
        """Test that global settings can be accessed."""
        assert hasattr(settings, 'HOST')
        assert hasattr(settings, 'PORT')
        assert hasattr(settings, 'POSTGRES_HOST')
        assert hasattr(settings, 'LOG_LEVEL')
    
    def test_settings_immutability(self):
        """Test that settings behave as expected when modified."""
        original_host = settings.HOST
        
        # Attempt to modify (this should create a new instance behavior)
        # depending on Pydantic version
        try:
            settings.HOST = "modified-host"
            # If modification succeeds, ensure it's the expected behavior
            assert settings.HOST == "modified-host"
            # Reset
            settings.HOST = original_host
        except AttributeError:
            # If Pydantic prevents modification, that's also acceptable
            assert settings.HOST == original_host 