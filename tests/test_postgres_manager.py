"""
Unit tests for PostgreSQL database manager.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import psycopg2
from contextlib import contextmanager

from src.snomed_ct_platform.database.postgres_manager import PostgresManager


class TestPostgresManagerInitialization:
    """Test cases for PostgresManager initialization."""
    
    def test_manager_initialization_with_defaults(self):
        """Test PostgresManager initialization with default parameters."""
        manager = PostgresManager()
        
        assert manager.host == "localhost"
        assert manager.port == 5432
        assert manager.database == "snomed_ct"
        assert manager.user == "postgres"
        assert manager.password == "postgres"
        assert manager.connection is None
        assert manager.connection_pool is None
    
    def test_manager_initialization_with_custom_params(self):
        """Test PostgresManager initialization with custom parameters."""
        manager = PostgresManager(
            host="custom-host",
            port=5433,
            database="custom_db",
            user="custom_user",
            password="custom_pass"
        )
        
        assert manager.host == "custom-host"
        assert manager.port == 5433
        assert manager.database == "custom_db"
        assert manager.user == "custom_user"
        assert manager.password == "custom_pass"
    
    def test_connection_string_generation(self):
        """Test database connection string generation."""
        manager = PostgresManager(
            host="testhost",
            port=5433,
            database="testdb",
            user="testuser",
            password="testpass"
        )
        
        conn_string = manager._get_connection_string()
        
        assert "host=testhost" in conn_string
        assert "port=5433" in conn_string
        assert "dbname=testdb" in conn_string
        assert "user=testuser" in conn_string
        assert "password=testpass" in conn_string


class TestConnectionManagement:
    """Test cases for connection management."""
    
    @patch('psycopg2.connect')
    def test_connect_success(self, mock_connect):
        """Test successful database connection."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        manager = PostgresManager()
        result = manager.connect()
        
        assert result is True
        assert manager.connection == mock_connection
        mock_connect.assert_called_once()
    
    @patch('psycopg2.connect')
    def test_connect_failure(self, mock_connect):
        """Test database connection failure."""
        mock_connect.side_effect = psycopg2.Error("Connection failed")
        
        manager = PostgresManager()
        result = manager.connect()
        
        assert result is False
        assert manager.connection is None
    
    def test_disconnect_with_connection(self):
        """Test disconnection when connection exists."""
        manager = PostgresManager()
        mock_connection = MagicMock()
        manager.connection = mock_connection
        
        manager.disconnect()
        
        mock_connection.close.assert_called_once()
        assert manager.connection is None
    
    def test_disconnect_without_connection(self):
        """Test disconnection when no connection exists."""
        manager = PostgresManager()
        
        # Should not raise an exception
        manager.disconnect()
        assert manager.connection is None
    
    @patch('psycopg2.connect')
    def test_check_connection_healthy(self, mock_connect):
        """Test connection health check when connection is healthy."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = None
        mock_connect.return_value = mock_connection
        
        manager = PostgresManager()
        manager.connect()
        
        result = manager.check_connection()
        
        assert result is True
        mock_cursor.execute.assert_called_with("SELECT 1")
    
    @patch('psycopg2.connect')
    def test_check_connection_unhealthy(self, mock_connect):
        """Test connection health check when connection is unhealthy."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Connection lost")
        mock_connect.return_value = mock_connection
        
        manager = PostgresManager()
        manager.connect()
        
        result = manager.check_connection()
        
        assert result is False
    
    def test_check_connection_no_connection(self):
        """Test connection health check when no connection exists."""
        manager = PostgresManager()
        
        result = manager.check_connection()
        
        assert result is False


class TestConceptOperations:
    """Test cases for concept-related database operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_get_concept_by_id_found(self):
        """Test retrieving concept by ID when concept exists."""
        # Mock database response
        mock_concept_data = (
            12345, "20220131", True, 900000000000207008, 900000000000074008
        )
        self.mock_cursor.fetchone.return_value = mock_concept_data
        
        concept = self.manager.get_concept_by_id(12345)
        
        assert concept is not None
        assert concept['id'] == 12345
        assert concept['effective_time'] == "20220131" 
        assert concept['active'] is True
        
        # Verify SQL query
        self.mock_cursor.execute.assert_called_once()
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "SELECT" in sql_call.upper()
        assert "concepts" in sql_call.lower()
        assert "WHERE id = %s" in sql_call
    
    def test_get_concept_by_id_not_found(self):
        """Test retrieving concept by ID when concept doesn't exist."""
        self.mock_cursor.fetchone.return_value = None
        
        concept = self.manager.get_concept_by_id(99999)
        
        assert concept is None
    
    def test_get_concepts_with_pagination(self):
        """Test retrieving concepts with pagination."""
        # Mock database response
        mock_concepts = [
            (12345, "20220131", True, 900000000000207008, 900000000000074008),
            (67890, "20220131", True, 900000000000207008, 900000000000073002)
        ]
        self.mock_cursor.fetchall.return_value = mock_concepts
        
        concepts = self.manager.get_concepts(page=1, page_size=10)
        
        assert len(concepts) == 2
        assert concepts[0]['id'] == 12345
        assert concepts[1]['id'] == 67890
        
        # Verify pagination in SQL
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "LIMIT" in sql_call.upper()
        assert "OFFSET" in sql_call.upper()
    
    def test_get_concepts_active_only(self):
        """Test retrieving only active concepts."""
        mock_concepts = [
            (12345, "20220131", True, 900000000000207008, 900000000000074008)
        ]
        self.mock_cursor.fetchall.return_value = mock_concepts
        
        concepts = self.manager.get_concepts(active_only=True)
        
        assert len(concepts) == 1
        assert concepts[0]['active'] is True
        
        # Verify WHERE clause for active concepts
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "WHERE active = true" in sql_call.lower()
    
    def test_search_concepts_by_term(self):
        """Test searching concepts by term."""
        mock_results = [
            (12345, "Heart disease", "Cardiac disorder description"),
            (67890, "Heart attack", "Myocardial infarction")
        ]
        self.mock_cursor.fetchall.return_value = mock_results
        
        results = self.manager.search_concepts_by_term("heart")
        
        assert len(results) == 2
        assert results[0]['concept_id'] == 12345
        assert "Heart disease" in results[0]['term']
        
        # Verify search query
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "ILIKE" in sql_call.upper() or "LIKE" in sql_call.upper()
    
    def test_insert_concept(self):
        """Test inserting a new concept."""
        concept_data = {
            'id': 12345,
            'effective_time': '20220131',
            'active': True,
            'module_id': 900000000000207008,
            'definition_status_id': 900000000000074008
        }
        
        result = self.manager.insert_concept(concept_data)
        
        assert result is True
        
        # Verify INSERT query
        self.mock_cursor.execute.assert_called_once()
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO" in sql_call.upper()
        assert "concepts" in sql_call.lower()
    
    def test_update_concept(self):
        """Test updating an existing concept."""
        concept_data = {
            'effective_time': '20220131',
            'active': False,
            'module_id': 900000000000207008
        }
        
        result = self.manager.update_concept(12345, concept_data)
        
        assert result is True
        
        # Verify UPDATE query
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "UPDATE" in sql_call.upper()
        assert "SET" in sql_call.upper()
        assert "WHERE id = %s" in sql_call


class TestDescriptionOperations:
    """Test cases for description-related database operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_get_descriptions_for_concept(self):
        """Test retrieving descriptions for a concept."""
        mock_descriptions = [
            (12345001, "20220131", True, 900000000000207008, 12345, 
             "en", 900000000000013009, "Heart disease", 900000000000448009),
            (12345002, "20220131", True, 900000000000207008, 12345,
             "en", 900000000000003001, "Cardiac disorder", 900000000000448009)
        ]
        self.mock_cursor.fetchall.return_value = mock_descriptions
        
        descriptions = self.manager.get_descriptions_for_concept(12345)
        
        assert len(descriptions) == 2
        assert descriptions[0]['concept_id'] == 12345
        assert descriptions[0]['term'] == "Heart disease"
        assert descriptions[1]['term'] == "Cardiac disorder"
    
    def test_get_descriptions_by_language(self):
        """Test retrieving descriptions filtered by language."""
        mock_descriptions = [
            (12345001, "20220131", True, 900000000000207008, 12345,
             "en", 900000000000013009, "Heart disease", 900000000000448009)
        ]
        self.mock_cursor.fetchall.return_value = mock_descriptions
        
        descriptions = self.manager.get_descriptions_for_concept(12345, language="en")
        
        assert len(descriptions) == 1
        assert descriptions[0]['language_code'] == "en"
        
        # Verify language filter in SQL
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "language_code = %s" in sql_call.lower()
    
    def test_insert_description(self):
        """Test inserting a new description."""
        description_data = {
            'id': 12345001,
            'effective_time': '20220131',
            'active': True,
            'module_id': 900000000000207008,
            'concept_id': 12345,
            'language_code': 'en',
            'type_id': 900000000000013009,
            'term': 'Heart disease',
            'case_significance_id': 900000000000448009
        }
        
        result = self.manager.insert_description(description_data) 
        
        assert result is True
        
        # Verify INSERT query
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO" in sql_call.upper()
        assert "descriptions" in sql_call.lower()


class TestRelationshipOperations:
    """Test cases for relationship-related database operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_get_relationships_for_concept(self):
        """Test retrieving relationships for a concept."""
        mock_relationships = [
            (12345001, "20220131", True, 900000000000207008, 12345, 404684003,
             0, 116680003, 900000000000011006, 900000000000451002)
        ]
        self.mock_cursor.fetchall.return_value = mock_relationships
        
        relationships = self.manager.get_relationships_for_concept(12345)
        
        assert len(relationships) == 1
        assert relationships[0]['source_id'] == 12345
        assert relationships[0]['destination_id'] == 404684003
    
    def test_get_relationships_by_type(self):
        """Test retrieving relationships filtered by type."""
        mock_relationships = [
            (12345001, "20220131", True, 900000000000207008, 12345, 404684003,
             0, 116680003, 900000000000011006, 900000000000451002)
        ]
        self.mock_cursor.fetchall.return_value = mock_relationships
        
        relationships = self.manager.get_relationships_for_concept(
            12345, relationship_type=116680003
        )
        
        assert len(relationships) == 1
        assert relationships[0]['type_id'] == 116680003
    
    def test_insert_relationship(self):
        """Test inserting a new relationship."""
        relationship_data = {
            'id': 12345001,
            'effective_time': '20220131',
            'active': True,
            'module_id': 900000000000207008, 
            'source_id': 12345,
            'destination_id': 404684003,
            'relationship_group': 0,
            'type_id': 116680003,
            'characteristic_type_id': 900000000000011006,
            'modifier_id': 900000000000451002
        }
        
        result = self.manager.insert_relationship(relationship_data)
        
        assert result is True
        
        # Verify INSERT query
        sql_call = self.mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO" in sql_call.upper()
        assert "relationships" in sql_call.lower()


class TestBulkOperations:
    """Test cases for bulk database operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_bulk_insert_concepts(self):
        """Test bulk inserting concepts."""
        concepts = [
            {
                'id': 12345,
                'effective_time': '20220131',
                'active': True,
                'module_id': 900000000000207008,
                'definition_status_id': 900000000000074008
            },
            {
                'id': 67890,
                'effective_time': '20220131', 
                'active': True,
                'module_id': 900000000000207008,
                'definition_status_id': 900000000000073002
            }
        ]
        
        result = self.manager.bulk_insert_concepts(concepts)
        
        assert result is True
        
        # Verify executemany was called
        self.mock_cursor.executemany.assert_called_once()
    
    def test_bulk_insert_descriptions(self):
        """Test bulk inserting descriptions."""
        descriptions = [
            {
                'id': 12345001,
                'effective_time': '20220131',
                'active': True,
                'module_id': 900000000000207008,
                'concept_id': 12345,
                'language_code': 'en',
                'type_id': 900000000000013009,
                'term': 'Heart disease',
                'case_significance_id': 900000000000448009
            }
        ]
        
        result = self.manager.bulk_insert_descriptions(descriptions)
        
        assert result is True
        self.mock_cursor.executemany.assert_called_once()
    
    def test_bulk_insert_relationships(self):
        """Test bulk inserting relationships."""
        relationships = [
            {
                'id': 12345001,
                'effective_time': '20220131',
                'active': True,
                'module_id': 900000000000207008,
                'source_id': 12345,
                'destination_id': 404684003,
                'relationship_group': 0,
                'type_id': 116680003,
                'characteristic_type_id': 900000000000011006,
                'modifier_id': 900000000000451002
            }
        ]
        
        result = self.manager.bulk_insert_relationships(relationships)
        
        assert result is True
        self.mock_cursor.executemany.assert_called_once()


class TestTransactionManagement:
    """Test cases for transaction management."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_transaction_commit_on_success(self):
        """Test transaction commit on successful operation."""
        concept_data = {
            'id': 12345,
            'effective_time': '20220131',
            'active': True,
            'module_id': 900000000000207008,
            'definition_status_id': 900000000000074008
        }
        
        self.manager.insert_concept(concept_data)
        
        # Should commit transaction
        self.mock_connection.commit.assert_called_once()
    
    def test_transaction_rollback_on_error(self):
        """Test transaction rollback on error."""
        self.mock_cursor.execute.side_effect = psycopg2.Error("Database error")
        
        concept_data = {
            'id': 12345,
            'effective_time': '20220131',
            'active': True,
            'module_id': 900000000000207008,
            'definition_status_id': 900000000000074008
        }
        
        result = self.manager.insert_concept(concept_data)
        
        assert result is False
        # Should rollback transaction
        self.mock_connection.rollback.assert_called_once()


class TestErrorHandling:
    """Test cases for error handling."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = PostgresManager()
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager.connection = self.mock_connection
    
    def test_database_error_handling(self):
        """Test handling of database errors."""
        self.mock_cursor.execute.side_effect = psycopg2.DatabaseError("Database error")
        
        concept = self.manager.get_concept_by_id(12345)
        
        assert concept is None
    
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        self.mock_cursor.execute.side_effect = psycopg2.OperationalError("Connection lost")
        
        result = self.manager.check_connection()
        
        assert result is False
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        # This test ensures parameterized queries are used
        malicious_input = "12345; DROP TABLE concepts; --"
        
        self.manager.get_concept_by_id(malicious_input)
        
        # Verify parameterized query was used
        call_args = self.mock_cursor.execute.call_args
        assert len(call_args) == 2  # SQL and parameters
        assert malicious_input in call_args[1]  # Parameter value 