"""
Pytest configuration and shared fixtures for SNOMED-CT platform tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any
import pandas as pd
import os

# Test data fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "HOST": "localhost",
        "PORT": 8000,
        "DEBUG": True,
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": 5432,
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_pass",
        "POSTGRES_DB": "test_db",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": 19530,
        "JANUSGRAPH_HOST": "localhost",
        "JANUSGRAPH_PORT": 8182,
        "LOG_LEVEL": "DEBUG"
    }

@pytest.fixture
def sample_concept_data() -> pd.DataFrame:
    """Sample concept data for testing."""
    return pd.DataFrame({
        'id': [12345, 67890, 11111],
        'effectiveTime': ['20220131', '20220131', '20220131'],
        'active': ['1', '1', '0'],
        'moduleId': [900000000000207008, 900000000000207008, 900000000000207008],
        'definitionStatusId': [900000000000074008, 900000000000073002, 900000000000074008]
    })

@pytest.fixture
def sample_description_data() -> pd.DataFrame:
    """Sample description data for testing."""
    return pd.DataFrame({
        'id': [12345001, 12345002, 67890001],
        'effectiveTime': ['20220131', '20220131', '20220131'],
        'active': ['1', '1', '1'],
        'moduleId': [900000000000207008, 900000000000207008, 900000000000207008],
        'conceptId': [12345, 12345, 67890],
        'languageCode': ['en', 'en', 'en'],
        'typeId': [900000000000013009, 900000000000003001, 900000000000013009],
        'term': ['Heart disease', 'Cardiac disorder', 'Diabetes mellitus'],
        'caseSignificanceId': [900000000000448009, 900000000000448009, 900000000000448009]
    })

@pytest.fixture
def sample_relationship_data() -> pd.DataFrame:
    """Sample relationship data for testing."""
    return pd.DataFrame({
        'id': [12345001, 12345002, 67890001],
        'effectiveTime': ['20220131', '20220131', '20220131'],
        'active': ['1', '1', '1'],
        'moduleId': [900000000000207008, 900000000000207008, 900000000000207008],
        'sourceId': [12345, 12345, 67890],
        'destinationId': [404684003, 404684003, 73211009],
        'relationshipGroup': [0, 0, 0],
        'typeId': [116680003, 116680003, 116680003],
        'characteristicTypeId': [900000000000011006, 900000000000011006, 900000000000011006],
        'modifierId': [900000000000451002, 900000000000451002, 900000000000451002]
    })

@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    return mock_conn

@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client."""
    mock_client = Mock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = []
    mock_client.insert.return_value = Mock(insert_count=1)
    return mock_client

@pytest.fixture
def mock_gremlin_client():
    """Mock Gremlin client."""
    mock_client = Mock()
    mock_traversal = Mock()
    mock_client.traversal.return_value = mock_traversal
    mock_traversal.V.return_value = mock_traversal
    mock_traversal.hasLabel.return_value = mock_traversal
    mock_traversal.toList.return_value = []
    return mock_client

@pytest.fixture
def sample_rf2_files(temp_dir) -> Dict[str, Path]:
    """Create sample RF2 files for testing."""
    files = {}
    
    # Create concept file
    concept_file = temp_dir / "sct2_Concept_Snapshot_INT_20220131.txt"
    concept_data = """id	effectiveTime	active	moduleId	definitionStatusId
12345	20220131	1	900000000000207008	900000000000074008
67890	20220131	1	900000000000207008	900000000000073002
11111	20220131	0	900000000000207008	900000000000074008"""
    concept_file.write_text(concept_data)
    files['concepts'] = concept_file
    
    # Create description file
    desc_file = temp_dir / "sct2_Description_Snapshot-en_INT_20220131.txt"
    desc_data = """id	effectiveTime	active	moduleId	conceptId	languageCode	typeId	term	caseSignificanceId
12345001	20220131	1	900000000000207008	12345	en	900000000000013009	Heart disease	900000000000448009
12345002	20220131	1	900000000000207008	12345	en	900000000000003001	Cardiac disorder	900000000000448009
67890001	20220131	1	900000000000207008	67890	en	900000000000013009	Diabetes mellitus	900000000000448009"""
    desc_file.write_text(desc_data)
    files['descriptions'] = desc_file
    
    # Create relationship file
    rel_file = temp_dir / "sct2_Relationship_Snapshot_INT_20220131.txt"
    rel_data = """id	effectiveTime	active	moduleId	sourceId	destinationId	relationshipGroup	typeId	characteristicTypeId	modifierId
12345001	20220131	1	900000000000207008	12345	404684003	0	116680003	900000000000011006	900000000000451002
12345002	20220131	1	900000000000207008	12345	404684003	0	116680003	900000000000011006	900000000000451002
67890001	20220131	1	900000000000207008	67890	73211009	0	116680003	900000000000011006	900000000000451002"""
    rel_file.write_text(rel_data)
    files['relationships'] = rel_file
    
    return files

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    test_env = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_pass",
        "POSTGRES_DB": "test_db",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530",
        "JANUSGRAPH_HOST": "localhost",
        "JANUSGRAPH_PORT": "8182",
        "LOG_LEVEL": "DEBUG",
        "DEBUG": "true"
    }
    
    # Set environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Clean up environment variables
    for key in test_env.keys():
        os.environ.pop(key, None) 