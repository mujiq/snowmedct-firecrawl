"""
Graph Ingestion Pipeline for SNOMED-CT Platform

This module orchestrates the complete graph ingestion process,
integrating PostgreSQL data with JanusGraph for relationship modeling.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .janusgraph_manager import JanusGraphManager, ConceptVertex, RelationshipEdge, RelationshipType
from ..database.postgres_manager import PostgresManager
from ..parsers.rf2_parser import ConceptRecord, RelationshipRecord
from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class GraphStats:
    """Statistics for graph ingestion pipeline."""
    start_time: float
    end_time: Optional[float] = None
    concepts_processed: int = 0
    concepts_created: int = 0
    relationships_processed: int = 0
    relationships_created: int = 0
    errors: int = 0
    batches_processed: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get pipeline duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def concepts_per_second(self) -> Optional[float]:
        """Get concepts processed per second."""
        if self.duration and self.duration > 0:
            return self.concepts_processed / self.duration
        return None


class GraphIngestionPipeline:
    """Pipeline for ingesting SNOMED-CT data into JanusGraph."""
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        concept_limit: Optional[int] = None,
        relationship_limit: Optional[int] = None
    ):
        """
        Initialize the graph ingestion pipeline.
        
        Args:
            batch_size: Number of records to process in each batch
            concept_limit: Maximum number of concepts to process
            relationship_limit: Maximum number of relationships to process
        """
        self.batch_size = batch_size or settings.application.batch_size
        self.concept_limit = concept_limit
        self.relationship_limit = relationship_limit
        
        # Initialize components
        self.janusgraph_manager = JanusGraphManager()
        self.postgres_manager = PostgresManager()
        
        # Pipeline state
        self.janusgraph_connected = False
        self.postgres_connected = False
        
        # Statistics
        self.stats = GraphStats(start_time=time.time())
        
        logger.info("Initialized GraphIngestionPipeline")
        logger.info(f"Batch size: {self.batch_size}")
        if self.concept_limit:
            logger.info(f"Concept limit: {self.concept_limit}")
        if self.relationship_limit:
            logger.info(f"Relationship limit: {self.relationship_limit}")
    
    def setup_connections(self) -> None:
        """Set up database connections."""
        try:
            logger.info("Setting up database connections")
            
            # Connect to JanusGraph
            logger.info("Connecting to JanusGraph...")
            self.janusgraph_manager.connect()
            self.janusgraph_connected = True
            
            # Connect to PostgreSQL
            logger.info("Connecting to PostgreSQL...")
            self.postgres_manager.connect()
            self.postgres_connected = True
            
            # Create JanusGraph schema
            logger.info("Setting up JanusGraph schema...")
            self.janusgraph_manager.create_schema()
            
            logger.info("Database connections established successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up connections: {e}")
            raise
    
    def get_concepts_from_postgres(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get active concepts from PostgreSQL.
        
        Args:
            limit: Maximum number of concepts to retrieve
            
        Returns:
            List of concept dictionaries
        """
        if not self.postgres_connected:
            raise ValueError("PostgreSQL not connected. Call setup_connections() first.")
        
        try:
            logger.info("Retrieving concepts from PostgreSQL")
            limit = limit or self.concept_limit
            concepts = self.postgres_manager.get_active_concepts(limit=limit)
            
            logger.info(f"Retrieved {len(concepts)} concepts from PostgreSQL")
            return concepts
            
        except Exception as e:
            logger.error(f"Failed to get concepts from PostgreSQL: {e}")
            raise
    
    def get_relationships_from_postgres(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get active relationships from PostgreSQL.
        
        Args:
            limit: Maximum number of relationships to retrieve
            
        Returns:
            List of relationship dictionaries
        """
        if not self.postgres_connected:
            raise ValueError("PostgreSQL not connected. Call setup_connections() first.")
        
        try:
            logger.info("Retrieving relationships from PostgreSQL")
            
            # Build query for active relationships
            query = """
                SELECT id, effective_time, active, module_id, source_id, 
                       destination_id, relationship_group, type_id, 
                       characteristic_type_id, modifier_id
                FROM relationships 
                WHERE active = true
            """
            
            if limit or self.relationship_limit:
                query += f" LIMIT {limit or self.relationship_limit}"
            
            with self.postgres_manager.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(query))
                relationships = [dict(row) for row in result]
            
            logger.info(f"Retrieved {len(relationships)} relationships from PostgreSQL")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get relationships from PostgreSQL: {e}")
            raise
    
    def convert_concepts_to_vertices(self, concepts: List[Dict[str, Any]]) -> List[ConceptVertex]:
        """
        Convert PostgreSQL concept data to JanusGraph vertices.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            List of ConceptVertex objects
        """
        try:
            vertices = []
            
            for concept in concepts:
                vertex = ConceptVertex(
                    concept_id=concept['id'],
                    fully_specified_name=concept.get('fully_specified_name', ''),
                    active=concept.get('active', True),
                    module_id=concept.get('module_id', 0),
                    definition_status_id=concept.get('definition_status_id', 0)
                )
                vertices.append(vertex)
            
            logger.debug(f"Converted {len(vertices)} concepts to vertices")
            return vertices
            
        except Exception as e:
            logger.error(f"Failed to convert concepts to vertices: {e}")
            raise
    
    def convert_relationships_to_edges(self, relationships: List[Dict[str, Any]]) -> List[RelationshipEdge]:
        """
        Convert PostgreSQL relationship data to JanusGraph edges.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            List of RelationshipEdge objects
        """
        try:
            edges = []
            
            for relationship in relationships:
                edge = RelationshipEdge(
                    relationship_id=relationship['id'],
                    source_id=relationship['source_id'],
                    destination_id=relationship['destination_id'],
                    type_id=relationship['type_id'],
                    relationship_group=relationship['relationship_group'],
                    active=relationship.get('active', True),
                    characteristic_type_id=relationship['characteristic_type_id'],
                    modifier_id=relationship['modifier_id']
                )
                edges.append(edge)
            
            logger.debug(f"Converted {len(edges)} relationships to edges")
            return edges
            
        except Exception as e:
            logger.error(f"Failed to convert relationships to edges: {e}")
            raise
    
    def ingest_concepts(self, concepts: List[Dict[str, Any]]) -> int:
        """
        Ingest concepts into JanusGraph.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Number of concepts created
        """
        if not self.janusgraph_connected:
            raise ValueError("JanusGraph not connected. Call setup_connections() first.")
        
        try:
            logger.info(f"Ingesting {len(concepts)} concepts into JanusGraph")
            
            # Convert to vertices
            vertices = self.convert_concepts_to_vertices(concepts)
            
            # Create vertices in batches
            created_vertices = self.janusgraph_manager.create_concepts_batch(
                vertices, 
                batch_size=self.batch_size
            )
            
            # Update statistics
            self.stats.concepts_processed += len(concepts)
            self.stats.concepts_created += len(created_vertices)
            
            logger.info(f"Successfully created {len(created_vertices)} concept vertices")
            return len(created_vertices)
            
        except Exception as e:
            logger.error(f"Failed to ingest concepts: {e}")
            self.stats.errors += 1
            raise
    
    def ingest_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Ingest relationships into JanusGraph.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Number of relationships created
        """
        if not self.janusgraph_connected:
            raise ValueError("JanusGraph not connected. Call setup_connections() first.")
        
        try:
            logger.info(f"Ingesting {len(relationships)} relationships into JanusGraph")
            
            # Convert to edges
            edges = self.convert_relationships_to_edges(relationships)
            
            # Create edges in batches
            created_edges = self.janusgraph_manager.create_relationships_batch(
                edges, 
                batch_size=self.batch_size
            )
            
            # Update statistics
            self.stats.relationships_processed += len(relationships)
            self.stats.relationships_created += len(created_edges)
            
            logger.info(f"Successfully created {len(created_edges)} relationship edges")
            return len(created_edges)
            
        except Exception as e:
            logger.error(f"Failed to ingest relationships: {e}")
            self.stats.errors += 1
            raise
    
    def run_graph_ingestion(
        self,
        ingest_concepts: bool = True,
        ingest_relationships: bool = True,
        test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete graph ingestion pipeline.
        
        Args:
            ingest_concepts: Whether to ingest concepts
            ingest_relationships: Whether to ingest relationships
            test_mode: Whether to run in test mode (small sample)
            
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting graph ingestion pipeline")
        
        if test_mode:
            # Use small limits for testing
            self.concept_limit = min(self.concept_limit or 100, 100)
            self.relationship_limit = min(self.relationship_limit or 200, 200)
            logger.info(f"Running in test mode - concepts: {self.concept_limit}, relationships: {self.relationship_limit}")
        
        try:
            # Set up connections
            self.setup_connections()
            
            concepts_created = 0
            relationships_created = 0
            
            # Ingest concepts first (vertices must exist before edges)
            if ingest_concepts:
                concepts = self.get_concepts_from_postgres()
                if concepts:
                    concepts_created = self.ingest_concepts(concepts)
                else:
                    logger.warning("No concepts found for ingestion")
            
            # Ingest relationships (edges)
            if ingest_relationships:
                relationships = self.get_relationships_from_postgres()
                if relationships:
                    relationships_created = self.ingest_relationships(relationships)
                else:
                    logger.warning("No relationships found for ingestion")
            
            # Update final statistics
            self.stats.end_time = time.time()
            
            # Get graph statistics
            graph_stats = self.janusgraph_manager.get_graph_statistics()
            
            # Create summary
            summary = {
                'success': True,
                'pipeline_stats': self._get_pipeline_stats(),
                'graph_stats': graph_stats,
                'ingestion_summary': {
                    'concepts_created': concepts_created,
                    'relationships_created': relationships_created
                }
            }
            
            logger.info(f"Graph ingestion pipeline completed successfully")
            logger.info(f"Concepts: {self.stats.concepts_processed} processed, {self.stats.concepts_created} created")
            logger.info(f"Relationships: {self.stats.relationships_processed} processed, {self.stats.relationships_created} created")
            logger.info(f"Duration: {self.stats.duration:.2f} seconds")
            
            return summary
            
        except Exception as e:
            self.stats.end_time = time.time()
            logger.error(f"Graph ingestion pipeline failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_stats': self._get_pipeline_stats()
            }
    
    def test_graph_operations(self) -> Dict[str, Any]:
        """
        Test graph operations and queries.
        
        Returns:
            Test results
        """
        logger.info("Testing graph operations")
        
        try:
            # Set up connections
            self.setup_connections()
            
            # Test JanusGraph operations
            janusgraph_test = self.janusgraph_manager.test_operations()
            
            # Run mini ingestion pipeline
            ingestion_test = self.run_graph_ingestion(test_mode=True)
            
            results = {
                'success': True,
                'janusgraph_test': janusgraph_test,
                'ingestion_test': ingestion_test,
                'connections_status': {
                    'janusgraph_connected': self.janusgraph_connected,
                    'postgres_connected': self.postgres_connected
                }
            }
            
            logger.info("Graph operations test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Graph operations test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def query_concept_hierarchy(self, concept_id: int, max_depth: int = 5) -> Dict[str, Any]:
        """
        Query concept hierarchy (parents and children).
        
        Args:
            concept_id: SNOMED-CT concept ID
            max_depth: Maximum traversal depth
            
        Returns:
            Hierarchy information
        """
        if not self.janusgraph_connected:
            raise ValueError("JanusGraph not connected. Call setup_connections() first.")
        
        try:
            # Get concept details
            concept = self.janusgraph_manager.get_concept_by_id(concept_id)
            
            if not concept:
                return {
                    'success': False,
                    'error': f'Concept {concept_id} not found'
                }
            
            # Get parents and children
            parents = self.janusgraph_manager.find_hierarchical_parents(concept_id, max_depth)
            children = self.janusgraph_manager.find_hierarchical_children(concept_id, max_depth)
            
            return {
                'success': True,
                'concept': concept,
                'parents': parents,
                'children': children,
                'parent_count': len(parents),
                'child_count': len(children)
            }
            
        except Exception as e:
            logger.error(f"Failed to query concept hierarchy for {concept_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'duration_seconds': self.stats.duration,
            'concepts_processed': self.stats.concepts_processed,
            'concepts_created': self.stats.concepts_created,
            'relationships_processed': self.stats.relationships_processed,
            'relationships_created': self.stats.relationships_created,
            'batches_processed': self.stats.batches_processed,
            'errors': self.stats.errors,
            'concepts_per_second': self.stats.concepts_per_second
        }
    
    def close_connections(self) -> None:
        """Close all database connections."""
        try:
            if self.janusgraph_connected:
                self.janusgraph_manager.close()
                self.janusgraph_connected = False
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


def main():
    """Main entry point for the graph ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SNOMED-CT Graph Ingestion Pipeline")
    parser.add_argument("--concept-limit", type=int, help="Maximum number of concepts to process")
    parser.add_argument("--relationship-limit", type=int, help="Maximum number of relationships to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--concepts-only", action="store_true", help="Ingest concepts only")
    parser.add_argument("--relationships-only", action="store_true", help="Ingest relationships only")
    parser.add_argument("--query-concept", type=int, help="Query hierarchy for specific concept ID")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GraphIngestionPipeline(
        batch_size=args.batch_size,
        concept_limit=args.concept_limit,
        relationship_limit=args.relationship_limit
    )
    
    try:
        # Handle different modes
        if args.query_concept:
            # Query specific concept hierarchy
            pipeline.setup_connections()
            result = pipeline.query_concept_hierarchy(args.query_concept)
            
            if result['success']:
                concept = result['concept']
                print(f"\n🔍 Concept: {concept['conceptId']} - {concept.get('fullySpecifiedName', 'N/A')}")
                print(f"   Parents: {result['parent_count']}")
                print(f"   Children: {result['child_count']}")
            else:
                print(f"\n❌ Query failed: {result['error']}")
            
            return 0 if result['success'] else 1
        
        elif args.test:
            # Test mode
            summary = pipeline.test_graph_operations()
        else:
            # Full ingestion
            summary = pipeline.run_graph_ingestion(
                ingest_concepts=not args.relationships_only,
                ingest_relationships=not args.concepts_only,
                test_mode=args.test
            )
        
        # Print summary
        if summary['success']:
            print("\n✅ Graph ingestion completed successfully!")
            if 'pipeline_stats' in summary:
                stats = summary['pipeline_stats']
                print(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")
                print(f"Concepts: {stats.get('concepts_created', 0)} created")
                print(f"Relationships: {stats.get('relationships_created', 0)} created")
        else:
            print(f"\n❌ Graph ingestion failed: {summary.get('error', 'Unknown error')}")
        
        return 0 if summary['success'] else 1
        
    finally:
        pipeline.close_connections()


if __name__ == "__main__":
    exit(main()) 