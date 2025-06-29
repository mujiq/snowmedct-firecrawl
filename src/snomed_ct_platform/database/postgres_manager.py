"""
PostgreSQL Database Manager for SNOMED-CT Platform

This module handles PostgreSQL database operations including schema creation,
data ingestion, and querying for SNOMED-CT data.
"""

from pathlib import Path
from typing import List, Dict, Optional, Generator, Any
import asyncio
from contextlib import asynccontextmanager

import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, BigInteger, String, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2.extras import execute_batch

from ..parsers.rf2_parser import ConceptRecord, DescriptionRecord, RelationshipRecord
from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

Base = declarative_base()


class PostgresManager:
    """Manages PostgreSQL database operations for SNOMED-CT data."""
    
    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize PostgreSQL manager.
        
        Args:
            connection_url: Database connection URL (uses settings if not provided)
        """
        self.connection_url = connection_url or settings.database.postgres_url
        self.engine = None
        self.Session = None
        
    def connect(self) -> None:
        """Establish database connection."""
        try:
            logger.info("Connecting to PostgreSQL database")
            self.engine = create_engine(
                self.connection_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Successfully connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def create_schema(self) -> None:
        """Create SNOMED-CT database schema."""
        logger.info("Creating SNOMED-CT database schema")
        
        try:
            with self.engine.begin() as conn:
                # Create concepts table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS concepts (
                        id BIGINT PRIMARY KEY,
                        effective_time VARCHAR(8) NOT NULL,
                        active BOOLEAN NOT NULL,
                        module_id BIGINT NOT NULL,
                        definition_status_id BIGINT NOT NULL
                    )
                """))
                
                # Create descriptions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS descriptions (
                        id BIGINT PRIMARY KEY,
                        effective_time VARCHAR(8) NOT NULL,
                        active BOOLEAN NOT NULL,
                        module_id BIGINT NOT NULL,
                        concept_id BIGINT NOT NULL,
                        language_code VARCHAR(2) NOT NULL,
                        type_id BIGINT NOT NULL,
                        term TEXT NOT NULL,
                        case_significance_id BIGINT NOT NULL,
                        FOREIGN KEY (concept_id) REFERENCES concepts(id)
                    )
                """))
                
                # Create relationships table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS relationships (
                        id BIGINT PRIMARY KEY,
                        effective_time VARCHAR(8) NOT NULL,
                        active BOOLEAN NOT NULL,
                        module_id BIGINT NOT NULL,
                        source_id BIGINT NOT NULL,
                        destination_id BIGINT NOT NULL,
                        relationship_group INTEGER NOT NULL,
                        type_id BIGINT NOT NULL,
                        characteristic_type_id BIGINT NOT NULL,
                        modifier_id BIGINT NOT NULL,
                        FOREIGN KEY (source_id) REFERENCES concepts(id),
                        FOREIGN KEY (destination_id) REFERENCES concepts(id)
                    )
                """))
                
                logger.info("Database schema created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise
    
    def create_indexes(self) -> None:
        """Create indexes for better query performance."""
        logger.info("Creating database indexes")
        
        try:
            with self.engine.begin() as conn:
                # Concepts indexes
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_concepts_active ON concepts(active)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_concepts_module_id ON concepts(module_id)"))
                
                # Descriptions indexes
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_descriptions_concept_id ON descriptions(concept_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_descriptions_active ON descriptions(active)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_descriptions_type_id ON descriptions(type_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_descriptions_term ON descriptions USING gin(to_tsvector('english', term))"))
                
                # Relationships indexes
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_relationships_source_id ON relationships(source_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_relationships_destination_id ON relationships(destination_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_relationships_type_id ON relationships(type_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_relationships_active ON relationships(active)"))
                
                logger.info("Database indexes created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
            raise
    
    def truncate_tables(self) -> None:
        """Truncate all SNOMED-CT tables."""
        logger.warning("Truncating all SNOMED-CT tables")
        
        try:
            with self.engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE relationships"))
                conn.execute(text("TRUNCATE TABLE descriptions"))
                conn.execute(text("TRUNCATE TABLE concepts"))
                
            logger.info("All tables truncated successfully")
            
        except Exception as e:
            logger.error(f"Failed to truncate tables: {e}")
            raise
    
    def insert_concepts_batch(self, concepts: List[ConceptRecord]) -> int:
        """
        Insert a batch of concept records.
        
        Args:
            concepts: List of concept records to insert
            
        Returns:
            Number of records inserted
        """
        if not concepts:
            return 0
        
        try:
            # Convert to list of tuples for batch insert
            data = [
                (c.id, c.effective_time, c.active, c.module_id, c.definition_status_id)
                for c in concepts
            ]
            
            # Use raw connection for better performance
            conn = psycopg2.connect(self.connection_url)
            cursor = conn.cursor()
            
            execute_batch(
                cursor,
                """
                INSERT INTO concepts (id, effective_time, active, module_id, definition_status_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    effective_time = EXCLUDED.effective_time,
                    active = EXCLUDED.active,
                    module_id = EXCLUDED.module_id,
                    definition_status_id = EXCLUDED.definition_status_id
                """,
                data,
                page_size=1000
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.debug(f"Inserted {len(concepts)} concept records")
            return len(concepts)
            
        except Exception as e:
            logger.error(f"Failed to insert concept batch: {e}")
            raise
    
    def insert_descriptions_batch(self, descriptions: List[DescriptionRecord]) -> int:
        """
        Insert a batch of description records.
        
        Args:
            descriptions: List of description records to insert
            
        Returns:
            Number of records inserted
        """
        if not descriptions:
            return 0
        
        try:
            # Convert to list of tuples for batch insert
            data = [
                (d.id, d.effective_time, d.active, d.module_id, d.concept_id,
                 d.language_code, d.type_id, d.term, d.case_significance_id)
                for d in descriptions
            ]
            
            # Use raw connection for better performance
            conn = psycopg2.connect(self.connection_url)
            cursor = conn.cursor()
            
            execute_batch(
                cursor,
                """
                INSERT INTO descriptions (id, effective_time, active, module_id, concept_id,
                                        language_code, type_id, term, case_significance_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    effective_time = EXCLUDED.effective_time,
                    active = EXCLUDED.active,
                    module_id = EXCLUDED.module_id,
                    concept_id = EXCLUDED.concept_id,
                    language_code = EXCLUDED.language_code,
                    type_id = EXCLUDED.type_id,
                    term = EXCLUDED.term,
                    case_significance_id = EXCLUDED.case_significance_id
                """,
                data,
                page_size=1000
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.debug(f"Inserted {len(descriptions)} description records")
            return len(descriptions)
            
        except Exception as e:
            logger.error(f"Failed to insert description batch: {e}")
            raise
    
    def insert_relationships_batch(self, relationships: List[RelationshipRecord]) -> int:
        """
        Insert a batch of relationship records.
        
        Args:
            relationships: List of relationship records to insert
            
        Returns:
            Number of records inserted
        """
        if not relationships:
            return 0
        
        try:
            # Convert to list of tuples for batch insert
            data = [
                (r.id, r.effective_time, r.active, r.module_id, r.source_id,
                 r.destination_id, r.relationship_group, r.type_id,
                 r.characteristic_type_id, r.modifier_id)
                for r in relationships
            ]
            
            # Use raw connection for better performance
            conn = psycopg2.connect(self.connection_url)
            cursor = conn.cursor()
            
            execute_batch(
                cursor,
                """
                INSERT INTO relationships (id, effective_time, active, module_id, source_id,
                                         destination_id, relationship_group, type_id,
                                         characteristic_type_id, modifier_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    effective_time = EXCLUDED.effective_time,
                    active = EXCLUDED.active,
                    module_id = EXCLUDED.module_id,
                    source_id = EXCLUDED.source_id,
                    destination_id = EXCLUDED.destination_id,
                    relationship_group = EXCLUDED.relationship_group,
                    type_id = EXCLUDED.type_id,
                    characteristic_type_id = EXCLUDED.characteristic_type_id,
                    modifier_id = EXCLUDED.modifier_id
                """,
                data,
                page_size=1000
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.debug(f"Inserted {len(relationships)} relationship records")
            return len(relationships)
            
        except Exception as e:
            logger.error(f"Failed to insert relationship batch: {e}")
            raise
    
    def get_active_concepts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get active concepts from the database.
        
        Args:
            limit: Maximum number of concepts to return
            
        Returns:
            List of concept dictionaries
        """
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT c.id, c.effective_time, c.active, c.module_id, c.definition_status_id,
                           d.term as fully_specified_name
                    FROM concepts c
                    LEFT JOIN descriptions d ON c.id = d.concept_id 
                    WHERE c.active = true 
                    AND d.active = true 
                    AND d.type_id = 900000000000003001  -- Fully Specified Name
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = conn.execute(text(query))
                concepts = [dict(row) for row in result]
                
                logger.info(f"Retrieved {len(concepts)} active concepts")
                return concepts
                
        except Exception as e:
            logger.error(f"Failed to get active concepts: {e}")
            raise
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables."""
        try:
            with self.engine.connect() as conn:
                counts = {}
                
                for table in ['concepts', 'descriptions', 'relationships']:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    counts[table] = result.scalar()
                
                logger.info(f"Table counts: {counts}")
                return counts
                
        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
            raise 