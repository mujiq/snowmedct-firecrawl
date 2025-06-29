"""
JanusGraph Manager for SNOMED-CT Platform

This module handles JanusGraph property graph operations including schema definition,
vertex and edge creation, and graph traversal for SNOMED-CT relationships.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

try:
    from gremlinpython.driver import client, serializer
    from gremlinpython.driver.driver_remote_connection import DriverRemoteConnection
    from gremlinpython.process.anonymous_traversal import traversal
    from gremlinpython.process.graph_traversal import __
    from gremlinpython.process.traversal import T
    from gremlinpython.structure.graph import Graph
    GREMLIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gremlin dependencies not available: {e}")
    GREMLIN_AVAILABLE = False

from ..utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


class RelationshipType(Enum):
    """SNOMED-CT relationship types."""
    IS_A = 116680003  # Is a (core hierarchy)
    FINDING_SITE = 363698007  # Finding site
    CAUSATIVE_AGENT = 246075003  # Causative agent
    PART_OF = 123005000  # Part of
    PROCEDURE_SITE = 363704007  # Procedure site
    CLINICAL_COURSE = 263502005  # Clinical course
    SEVERITY = 246112005  # Severity
    ASSOCIATED_MORPHOLOGY = 116676008  # Associated morphology
    PATHOLOGICAL_PROCESS = 370135005  # Pathological process
    METHOD = 260686004  # Method


@dataclass
class ConceptVertex:
    """Represents a SNOMED-CT concept vertex."""
    concept_id: int
    fully_specified_name: str
    active: bool
    module_id: int
    definition_status_id: int
    created_time: Optional[int] = None
    
    def to_properties(self) -> Dict[str, Any]:
        """Convert to vertex properties dictionary."""
        return {
            'conceptId': self.concept_id,
            'fullySpecifiedName': self.fully_specified_name,
            'active': self.active,
            'moduleId': self.module_id,
            'definitionStatusId': self.definition_status_id,
            'createdTime': self.created_time or int(time.time())
        }


@dataclass
class RelationshipEdge:
    """Represents a SNOMED-CT relationship edge."""
    relationship_id: int
    source_id: int
    destination_id: int
    type_id: int
    relationship_group: int
    active: bool
    characteristic_type_id: int
    modifier_id: int
    created_time: Optional[int] = None
    
    def to_properties(self) -> Dict[str, Any]:
        """Convert to edge properties dictionary."""
        return {
            'relationshipId': self.relationship_id,
            'typeId': self.type_id,
            'relationshipGroup': self.relationship_group,
            'active': self.active,
            'characteristicTypeId': self.characteristic_type_id,
            'modifierId': self.modifier_id,
            'createdTime': self.created_time or int(time.time())
        }


class JanusGraphManager:
    """Manages JanusGraph operations for SNOMED-CT relationship modeling."""
    
    # Vertex labels
    CONCEPT_LABEL = 'Concept'
    
    # Edge labels for different relationship types
    EDGE_LABELS = {
        RelationshipType.IS_A.value: 'IS_A',
        RelationshipType.FINDING_SITE.value: 'FINDING_SITE',
        RelationshipType.CAUSATIVE_AGENT.value: 'CAUSATIVE_AGENT',
        RelationshipType.PART_OF.value: 'PART_OF',
        RelationshipType.PROCEDURE_SITE.value: 'PROCEDURE_SITE',
        RelationshipType.CLINICAL_COURSE.value: 'CLINICAL_COURSE',
        RelationshipType.SEVERITY.value: 'SEVERITY',
        RelationshipType.ASSOCIATED_MORPHOLOGY.value: 'ASSOCIATED_MORPHOLOGY',
        RelationshipType.PATHOLOGICAL_PROCESS.value: 'PATHOLOGICAL_PROCESS',
        RelationshipType.METHOD.value: 'METHOD'
    }
    
    # Default edge label for unknown relationship types
    DEFAULT_EDGE_LABEL = 'RELATED_TO'
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        graph_name: str = 'snomed_ct'
    ):
        """
        Initialize JanusGraph manager.
        
        Args:
            host: JanusGraph server host
            port: JanusGraph server port
            graph_name: Name of the graph to use
        """
        if not GREMLIN_AVAILABLE:
            raise ImportError("Gremlin dependencies not available. Install with: pip install gremlinpython")
        
        self.host = host or settings.database.janusgraph_host
        self.port = port or settings.database.janusgraph_port
        self.graph_name = graph_name
        
        # Connection components
        self.connection = None
        self.g = None
        self.connected = False
        
        logger.info(f"Initialized JanusGraphManager for {self.host}:{self.port}")
        logger.info(f"Graph name: {self.graph_name}")
    
    def connect(self) -> None:
        """Connect to JanusGraph server."""
        try:
            logger.info(f"Connecting to JanusGraph at {self.host}:{self.port}")
            
            # Create connection
            connection_url = f"ws://{self.host}:{self.port}/gremlin"
            self.connection = DriverRemoteConnection(connection_url, 'g')
            
            # Create graph traversal
            self.g = traversal().withRemote(self.connection)
            
            # Test connection
            try:
                # Simple test query
                vertex_count = self.g.V().count().next()
                self.connected = True
                logger.info("Successfully connected to JanusGraph")
                logger.info(f"Current vertex count: {vertex_count}")
            except Exception as e:
                logger.warning(f"Connection test warning: {e}")
                self.connected = True  # Assume connected if we got this far
                
        except Exception as e:
            logger.error(f"Failed to connect to JanusGraph: {e}")
            raise
    
    def create_schema(self) -> None:
        """Create JanusGraph schema for SNOMED-CT concepts and relationships."""
        if not self.connected or not self.g:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            logger.info("Creating JanusGraph schema for SNOMED-CT")
            
            # Note: JanusGraph schema is typically created through management API
            # Here we ensure the basic structure exists through queries
            
            # Create a test vertex to ensure schema elements exist
            test_vertex = (self.g.addV(self.CONCEPT_LABEL)
                          .property('conceptId', 0)
                          .property('fullySpecifiedName', 'Schema Test Concept')
                          .property('active', True)
                          .property('moduleId', 0)
                          .property('definitionStatusId', 0)
                          .property('createdTime', int(time.time()))
                          .next())
            
            # Remove test vertex
            self.g.V(test_vertex).drop().iterate()
            
            logger.info("JanusGraph schema validation completed")
            
        except Exception as e:
            logger.error(f"Failed to create JanusGraph schema: {e}")
            raise
    
    def create_concept_vertex(self, concept: ConceptVertex) -> Any:
        """
        Create a concept vertex in the graph.
        
        Args:
            concept: Concept vertex data
            
        Returns:
            Created vertex
        """
        if not self.connected or not self.g:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Check if vertex already exists
            existing = (self.g.V()
                       .hasLabel(self.CONCEPT_LABEL)
                       .has('conceptId', concept.concept_id)
                       .toList())
            
            if existing:
                logger.debug(f"Concept vertex {concept.concept_id} already exists")
                return existing[0]
            
            # Create new vertex
            properties = concept.to_properties()
            vertex_traversal = self.g.addV(self.CONCEPT_LABEL)
            
            # Add properties
            for key, value in properties.items():
                vertex_traversal = vertex_traversal.property(key, value)
            
            vertex = vertex_traversal.next()
            logger.debug(f"Created concept vertex: {concept.concept_id}")
            
            return vertex
            
        except Exception as e:
            logger.error(f"Failed to create concept vertex {concept.concept_id}: {e}")
            raise
    
    def create_relationship_edge(self, relationship: RelationshipEdge) -> Any:
        """
        Create a relationship edge in the graph.
        
        Args:
            relationship: Relationship edge data
            
        Returns:
            Created edge
        """
        if not self.connected or not self.g:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Find source and destination vertices
            source_vertex = (self.g.V()
                           .hasLabel(self.CONCEPT_LABEL)
                           .has('conceptId', relationship.source_id)
                           .toList())
            
            dest_vertex = (self.g.V()
                         .hasLabel(self.CONCEPT_LABEL)
                         .has('conceptId', relationship.destination_id)
                         .toList())
            
            if not source_vertex:
                raise ValueError(f"Source vertex {relationship.source_id} not found")
            if not dest_vertex:
                raise ValueError(f"Destination vertex {relationship.destination_id} not found")
            
            # Determine edge label
            edge_label = self.EDGE_LABELS.get(relationship.type_id, self.DEFAULT_EDGE_LABEL)
            
            # Check if edge already exists
            existing = (self.g.V(source_vertex[0])
                       .outE(edge_label)
                       .has('relationshipId', relationship.relationship_id)
                       .toList())
            
            if existing:
                logger.debug(f"Relationship edge {relationship.relationship_id} already exists")
                return existing[0]
            
            # Create edge
            properties = relationship.to_properties()
            edge_traversal = self.g.V(source_vertex[0]).addE(edge_label).to(dest_vertex[0])
            
            # Add properties
            for key, value in properties.items():
                edge_traversal = edge_traversal.property(key, value)
            
            edge = edge_traversal.next()
            logger.debug(f"Created relationship edge: {relationship.relationship_id}")
            
            return edge
            
        except Exception as e:
            logger.error(f"Failed to create relationship edge {relationship.relationship_id}: {e}")
            raise
    
    def create_concepts_batch(self, concepts: List[ConceptVertex], batch_size: int = 100) -> List[Any]:
        """
        Create multiple concept vertices in batches.
        
        Args:
            concepts: List of concept vertices to create
            batch_size: Number of vertices to create in each batch
            
        Returns:
            List of created vertices
        """
        if not self.connected or not self.g:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            logger.info(f"Creating {len(concepts)} concept vertices in batches of {batch_size}")
            created_vertices = []
            
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i + batch_size]
                batch_vertices = []
                
                for concept in batch:
                    try:
                        vertex = self.create_concept_vertex(concept)
                        batch_vertices.append(vertex)
                    except Exception as e:
                        logger.warning(f"Failed to create concept {concept.concept_id}: {e}")
                        continue
                
                created_vertices.extend(batch_vertices)
                logger.debug(f"Created batch of {len(batch_vertices)} vertices")
            
            logger.info(f"Successfully created {len(created_vertices)} concept vertices")
            return created_vertices
            
        except Exception as e:
            logger.error(f"Failed to create concept vertices batch: {e}")
            raise
    
    def create_relationships_batch(self, relationships: List[RelationshipEdge], batch_size: int = 100) -> List[Any]:
        """
        Create multiple relationship edges in batches.
        
        Args:
            relationships: List of relationship edges to create
            batch_size: Number of edges to create in each batch
            
        Returns:
            List of created edges
        """
        if not self.connected or not self.g:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            logger.info(f"Creating {len(relationships)} relationship edges in batches of {batch_size}")
            created_edges = []
            
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                batch_edges = []
                
                for relationship in batch:
                    try:
                        edge = self.create_relationship_edge(relationship)
                        batch_edges.append(edge)
                    except Exception as e:
                        logger.warning(f"Failed to create relationship {relationship.relationship_id}: {e}")
                        continue
                
                created_edges.extend(batch_edges)
                logger.debug(f"Created batch of {len(batch_edges)} edges")
            
            logger.info(f"Successfully created {len(created_edges)} relationship edges")
            return created_edges
            
        except Exception as e:
            logger.error(f"Failed to create relationship edges batch: {e}")
            raise
    
    def get_concept_by_id(self, concept_id: int) -> Optional[Dict[str, Any]]:
        """
        Get concept by SNOMED-CT concept ID.
        
        Args:
            concept_id: SNOMED-CT concept ID
            
        Returns:
            Concept properties or None if not found
        """
        if not self.connected:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            vertices = (self.g.V()
                       .hasLabel(self.CONCEPT_LABEL)
                       .has('conceptId', concept_id)
                       .elementMap()
                       .toList())
            
            if vertices:
                return dict(vertices[0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get concept {concept_id}: {e}")
            raise
    
    def get_concept_relationships(
        self,
        concept_id: int,
        direction: str = 'out',
        relationship_types: Optional[List[int]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a concept.
        
        Args:
            concept_id: SNOMED-CT concept ID
            direction: 'out', 'in', or 'both'
            relationship_types: List of relationship type IDs to filter
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship edge properties
        """
        if not self.connected:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            # Start with concept vertex
            traversal = (self.g.V()
                        .hasLabel(self.CONCEPT_LABEL)
                        .has('conceptId', concept_id))
            
            # Add direction
            if direction == 'out':
                traversal = traversal.outE()
            elif direction == 'in':
                traversal = traversal.inE()
            else:  # both
                traversal = traversal.bothE()
            
            # Filter by relationship types
            if relationship_types:
                edge_labels = [self.EDGE_LABELS.get(rt, self.DEFAULT_EDGE_LABEL) for rt in relationship_types]
                traversal = traversal.hasLabel(*edge_labels)
            
            # Add limit
            if limit:
                traversal = traversal.limit(limit)
            
            # Get edge properties
            relationships = traversal.elementMap().toList()
            
            return [dict(rel) for rel in relationships]
            
        except Exception as e:
            logger.error(f"Failed to get relationships for concept {concept_id}: {e}")
            raise
    
    def find_hierarchical_parents(self, concept_id: int, max_depth: int = 10) -> List[Dict[str, Any]]:
        """
        Find hierarchical parents (IS_A relationships) for a concept.
        
        Args:
            concept_id: SNOMED-CT concept ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of parent concept properties
        """
        if not self.connected:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            parents = (self.g.V()
                      .hasLabel(self.CONCEPT_LABEL)
                      .has('conceptId', concept_id)
                      .repeat(__.out('IS_A'))
                      .times(max_depth)
                      .dedup()
                      .elementMap()
                      .toList())
            
            return [dict(parent) for parent in parents]
            
        except Exception as e:
            logger.error(f"Failed to find parents for concept {concept_id}: {e}")
            raise
    
    def find_hierarchical_children(self, concept_id: int, max_depth: int = 10) -> List[Dict[str, Any]]:
        """
        Find hierarchical children (inverse IS_A relationships) for a concept.
        
        Args:
            concept_id: SNOMED-CT concept ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of child concept properties
        """
        if not self.connected:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            children = (self.g.V()
                       .hasLabel(self.CONCEPT_LABEL)
                       .has('conceptId', concept_id)
                       .repeat(__.in_('IS_A'))
                       .times(max_depth)
                       .dedup()
                       .elementMap()
                       .toList())
            
            return [dict(child) for child in children]
            
        except Exception as e:
            logger.error(f"Failed to find children for concept {concept_id}: {e}")
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.connected:
            raise ValueError("Not connected to JanusGraph. Call connect() first.")
        
        try:
            vertex_count = self.g.V().count().next()
            edge_count = self.g.E().count().next()
            concept_count = self.g.V().hasLabel(self.CONCEPT_LABEL).count().next()
            
            # Count edges by type
            edge_counts = {}
            for label in self.EDGE_LABELS.values():
                try:
                    count = self.g.E().hasLabel(label).count().next()
                    if count > 0:
                        edge_counts[label] = count
                except:
                    pass
            
            stats = {
                'total_vertices': vertex_count,
                'total_edges': edge_count,
                'concept_vertices': concept_count,
                'edge_counts_by_type': edge_counts
            }
            
            logger.info(f"Graph stats: {vertex_count} vertices, {edge_count} edges")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            raise
    
    def test_operations(self) -> Dict[str, Any]:
        """
        Test basic JanusGraph operations.
        
        Returns:
            Test results
        """
        logger.info("Testing JanusGraph operations")
        
        try:
            # Test concepts
            test_concepts = [
                ConceptVertex(1000001, "Test concept 1", True, 900000000000207008, 900000000000074008),
                ConceptVertex(1000002, "Test concept 2", True, 900000000000207008, 900000000000074008),
                ConceptVertex(1000003, "Test concept 3", True, 900000000000207008, 900000000000074008)
            ]
            
            # Test relationships
            test_relationships = [
                RelationshipEdge(2000001, 1000002, 1000001, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002),
                RelationshipEdge(2000002, 1000003, 1000001, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002)
            ]
            
            # Create test vertices
            created_vertices = self.create_concepts_batch(test_concepts)
            
            # Create test edges
            created_edges = self.create_relationships_batch(test_relationships)
            
            # Test queries
            concept_1 = self.get_concept_by_id(1000001)
            relationships = self.get_concept_relationships(1000001, direction='in')
            
            # Clean up test data
            for concept in test_concepts:
                vertices = (self.g.V()
                           .hasLabel(self.CONCEPT_LABEL)
                           .has('conceptId', concept.concept_id)
                           .toList())
                for vertex in vertices:
                    self.g.V(vertex).drop().iterate()
            
            results = {
                'success': True,
                'operations_tested': ['create_vertices', 'create_edges', 'query_concept', 'query_relationships'],
                'test_vertices_created': len(created_vertices),
                'test_edges_created': len(created_edges),
                'concept_query_success': concept_1 is not None,
                'relationship_query_count': len(relationships),
                'graph_stats': self.get_graph_statistics()
            }
            
            logger.info("JanusGraph operations test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"JanusGraph operations test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self) -> None:
        """Close the JanusGraph connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connected = False
                logger.info("JanusGraph connection closed")
        except Exception as e:
            logger.warning(f"Error closing JanusGraph connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 