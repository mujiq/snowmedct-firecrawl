"""
SNOMED-CT Data Ingestion Pipeline

This module orchestrates the complete data ingestion process from RF2 file parsing
to multi-modal database storage across PostgreSQL, Milvus, and JanusGraph.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .parsers.rf2_parser import RF2Parser
from .database.postgres_manager import PostgresManager
from .utils.logging import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    start_time: float
    end_time: Optional[float] = None
    concepts_processed: int = 0
    descriptions_processed: int = 0
    relationships_processed: int = 0
    errors: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get pipeline duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def total_records(self) -> int:
        """Get total records processed."""
        return self.concepts_processed + self.descriptions_processed + self.relationships_processed


class SnomedCTPipeline:
    """Main SNOMED-CT data ingestion pipeline."""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        batch_size: Optional[int] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the SNOMED-CT pipeline.
        
        Args:
            data_dir: Directory containing SNOMED-CT data files
            batch_size: Number of records to process in each batch
            log_level: Logging level
        """
        self.data_dir = Path(data_dir) if data_dir else settings.application.data_dir
        self.batch_size = batch_size or settings.application.batch_size
        
        # Set up logging
        setup_logging(
            log_level=log_level,
            log_file=Path("logs/snomed_pipeline.log")
        )
        
        # Initialize components
        self.rf2_parser = RF2Parser(self.data_dir)
        self.postgres_manager = PostgresManager()
        
        # Pipeline statistics
        self.stats = PipelineStats(start_time=time.time())
        
        logger.info(f"Initialized SNOMED-CT pipeline with data_dir: {self.data_dir}")
    
    def setup_databases(self, recreate_schema: bool = False) -> None:
        """
        Set up database connections and schemas.
        
        Args:
            recreate_schema: Whether to recreate the database schema
        """
        logger.info("Setting up databases")
        
        try:
            # Setup PostgreSQL
            self.postgres_manager.connect()
            
            if recreate_schema:
                logger.warning("Recreating database schema - all data will be lost!")
                self.postgres_manager.truncate_tables()
            
            self.postgres_manager.create_schema()
            self.postgres_manager.create_indexes()
            
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup databases: {e}")
            raise
    
    def extract_and_find_files(self, archive_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Extract RF2 archive and find relevant files.
        
        Args:
            archive_path: Path to RF2 zip archive
            
        Returns:
            Dictionary mapping file types to file paths
        """
        logger.info("Extracting and finding RF2 files")
        
        try:
            if archive_path and archive_path.exists():
                logger.info(f"Extracting archive: {archive_path}")
                self.rf2_parser.extract_rf2_archive(archive_path)
            
            rf2_files = self.rf2_parser.find_rf2_files()
            
            if not rf2_files:
                raise ValueError("No RF2 files found in the specified directory")
            
            logger.info(f"Found {len(rf2_files)} RF2 files: {list(rf2_files.keys())}")
            return rf2_files
            
        except Exception as e:
            logger.error(f"Failed to extract and find files: {e}")
            raise
    
    def ingest_concepts(self, concepts_file: Path) -> int:
        """
        Ingest concepts from RF2 file to PostgreSQL.
        
        Args:
            concepts_file: Path to concepts RF2 file
            
        Returns:
            Number of concepts processed
        """
        logger.info(f"Ingesting concepts from: {concepts_file}")
        
        concepts_processed = 0
        
        try:
            # Get file statistics
            stats = self.rf2_parser.get_file_stats(concepts_file)
            logger.info(f"Concepts file contains {stats['total_records']} records")
            
            # Process concepts in batches
            for concept_batch in self.rf2_parser.parse_concepts(concepts_file, self.batch_size):
                try:
                    inserted = self.postgres_manager.insert_concepts_batch(concept_batch)
                    concepts_processed += inserted
                    
                    if concepts_processed % (self.batch_size * 10) == 0:
                        logger.info(f"Processed {concepts_processed} concepts so far...")
                        
                except Exception as e:
                    logger.error(f"Failed to insert concept batch: {e}")
                    self.stats.errors += 1
                    continue
            
            self.stats.concepts_processed = concepts_processed
            logger.info(f"Successfully ingested {concepts_processed} concepts")
            
            return concepts_processed
            
        except Exception as e:
            logger.error(f"Failed to ingest concepts: {e}")
            raise
    
    def ingest_descriptions(self, descriptions_file: Path) -> int:
        """
        Ingest descriptions from RF2 file to PostgreSQL.
        
        Args:
            descriptions_file: Path to descriptions RF2 file
            
        Returns:
            Number of descriptions processed
        """
        logger.info(f"Ingesting descriptions from: {descriptions_file}")
        
        descriptions_processed = 0
        
        try:
            # Get file statistics
            stats = self.rf2_parser.get_file_stats(descriptions_file)
            logger.info(f"Descriptions file contains {stats['total_records']} records")
            
            # Process descriptions in batches
            for description_batch in self.rf2_parser.parse_descriptions(descriptions_file, self.batch_size):
                try:
                    inserted = self.postgres_manager.insert_descriptions_batch(description_batch)
                    descriptions_processed += inserted
                    
                    if descriptions_processed % (self.batch_size * 10) == 0:
                        logger.info(f"Processed {descriptions_processed} descriptions so far...")
                        
                except Exception as e:
                    logger.error(f"Failed to insert description batch: {e}")
                    self.stats.errors += 1
                    continue
            
            self.stats.descriptions_processed = descriptions_processed
            logger.info(f"Successfully ingested {descriptions_processed} descriptions")
            
            return descriptions_processed
            
        except Exception as e:
            logger.error(f"Failed to ingest descriptions: {e}")
            raise
    
    def ingest_relationships(self, relationships_file: Path) -> int:
        """
        Ingest relationships from RF2 file to PostgreSQL.
        
        Args:
            relationships_file: Path to relationships RF2 file
            
        Returns:
            Number of relationships processed
        """
        logger.info(f"Ingesting relationships from: {relationships_file}")
        
        relationships_processed = 0
        
        try:
            # Get file statistics
            stats = self.rf2_parser.get_file_stats(relationships_file)
            logger.info(f"Relationships file contains {stats['total_records']} records")
            
            # Process relationships in batches
            for relationship_batch in self.rf2_parser.parse_relationships(relationships_file, self.batch_size):
                try:
                    inserted = self.postgres_manager.insert_relationships_batch(relationship_batch)
                    relationships_processed += inserted
                    
                    if relationships_processed % (self.batch_size * 10) == 0:
                        logger.info(f"Processed {relationships_processed} relationships so far...")
                        
                except Exception as e:
                    logger.error(f"Failed to insert relationship batch: {e}")
                    self.stats.errors += 1
                    continue
            
            self.stats.relationships_processed = relationships_processed
            logger.info(f"Successfully ingested {relationships_processed} relationships")
            
            return relationships_processed
            
        except Exception as e:
            logger.error(f"Failed to ingest relationships: {e}")
            raise
    
    def run_postgresql_ingestion(
        self,
        archive_path: Optional[Path] = None,
        recreate_schema: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete PostgreSQL ingestion pipeline.
        
        Args:
            archive_path: Path to RF2 zip archive
            recreate_schema: Whether to recreate the database schema
            
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting SNOMED-CT PostgreSQL ingestion pipeline")
        
        try:
            # Setup databases
            self.setup_databases(recreate_schema=recreate_schema)
            
            # Extract and find RF2 files
            rf2_files = self.extract_and_find_files(archive_path)
            
            # Ingest data in dependency order: concepts first, then descriptions, then relationships
            if 'concepts' in rf2_files:
                self.ingest_concepts(rf2_files['concepts'])
            else:
                logger.error("Concepts file not found - cannot proceed")
                raise ValueError("Concepts file is required for ingestion")
            
            if 'descriptions' in rf2_files:
                self.ingest_descriptions(rf2_files['descriptions'])
            else:
                logger.warning("Descriptions file not found - skipping")
            
            if 'relationships' in rf2_files:
                self.ingest_relationships(rf2_files['relationships'])
            else:
                logger.warning("Relationships file not found - skipping")
            
            # Get final database statistics
            table_counts = self.postgres_manager.get_table_counts()
            
            # Update pipeline statistics
            self.stats.end_time = time.time()
            
            # Create summary
            summary = {
                'success': True,
                'duration_seconds': self.stats.duration,
                'records_processed': {
                    'concepts': self.stats.concepts_processed,
                    'descriptions': self.stats.descriptions_processed,
                    'relationships': self.stats.relationships_processed,
                    'total': self.stats.total_records
                },
                'database_counts': table_counts,
                'errors': self.stats.errors
            }
            
            logger.info(f"Pipeline completed successfully in {self.stats.duration:.2f} seconds")
            logger.info(f"Total records processed: {self.stats.total_records}")
            logger.info(f"Database record counts: {table_counts}")
            
            return summary
            
        except Exception as e:
            self.stats.end_time = time.time()
            logger.error(f"Pipeline failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': self.stats.duration,
                'records_processed': {
                    'concepts': self.stats.concepts_processed,
                    'descriptions': self.stats.descriptions_processed,
                    'relationships': self.stats.relationships_processed,
                    'total': self.stats.total_records
                },
                'errors': self.stats.errors
            }


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SNOMED-CT Data Ingestion Pipeline")
    parser.add_argument("--data-dir", type=Path, help="Directory containing SNOMED-CT data")
    parser.add_argument("--archive", type=Path, help="Path to RF2 zip archive")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--recreate-schema", action="store_true", help="Recreate database schema")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SnomedCTPipeline(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        log_level=args.log_level
    )
    
    # Run pipeline
    summary = pipeline.run_postgresql_ingestion(
        archive_path=args.archive,
        recreate_schema=args.recreate_schema
    )
    
    # Print summary
    if summary['success']:
        print("\n✅ Pipeline completed successfully!")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Records processed: {summary['records_processed']['total']}")
        print(f"Database counts: {summary['database_counts']}")
    else:
        print(f"\n❌ Pipeline failed: {summary['error']}")
    
    return 0 if summary['success'] else 1


if __name__ == "__main__":
    exit(main()) 