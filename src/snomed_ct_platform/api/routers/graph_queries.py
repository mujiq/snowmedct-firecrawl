"""
Graph queries router for JanusGraph traversal and analysis.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..dependencies import get_janusgraph_manager, get_postgres_manager
from ..config import settings

router = APIRouter()


class GraphConceptResponse(BaseModel):
    """Response model for graph concept vertex."""
    
    concept_id: int = Field(..., description="SNOMED-CT concept ID")
    fully_specified_name: str = Field(..., description="Fully specified name")
    active: bool = Field(..., description="Whether the concept is active")
    module_id: int = Field(..., description="Module ID")
    definition_status_id: int = Field(..., description="Definition status ID")
    created_time: Optional[int] = Field(None, description="Creation timestamp")


class GraphRelationshipResponse(BaseModel):
    """Response model for graph relationship edge."""
    
    relationship_id: int = Field(..., description="Relationship ID")
    source_id: int = Field(..., description="Source concept ID")
    destination_id: int = Field(..., description="Destination concept ID")
    type_id: int = Field(..., description="Relationship type ID")
    relationship_group: int = Field(..., description="Relationship group")
    active: bool = Field(..., description="Whether the relationship is active")
    characteristic_type_id: int = Field(..., description="Characteristic type ID")
    modifier_id: int = Field(..., description="Modifier ID")
    edge_label: Optional[str] = Field(None, description="Edge label in graph")


class HierarchyTraversalRequest(BaseModel):
    """Request model for hierarchy traversal."""
    
    concept_id: int = Field(..., description="Starting concept ID")
    direction: str = Field("up", regex="^(up|down|both)$", description="Traversal direction")
    max_depth: int = Field(default=3, description="Maximum traversal depth", ge=1, le=10)
    active_only: bool = Field(default=True, description="Include only active concepts")


class HierarchyResponse(BaseModel):
    """Response model for hierarchy traversal."""
    
    root_concept_id: int = Field(..., description="Root concept ID")
    direction: str = Field(..., description="Traversal direction")
    max_depth: int = Field(..., description="Maximum depth requested")
    concepts: List[GraphConceptResponse] = Field(..., description="Found concepts")
    relationships: List[GraphRelationshipResponse] = Field(..., description="Found relationships")
    depth_reached: int = Field(..., description="Actual depth reached")


class PathFindingRequest(BaseModel):
    """Request model for path finding between concepts."""
    
    source_id: int = Field(..., description="Source concept ID")
    target_id: int = Field(..., description="Target concept ID")
    max_path_length: int = Field(default=5, description="Maximum path length", ge=1, le=10)
    relationship_types: Optional[List[int]] = Field(None, description="Allowed relationship types")


class ConceptPathResponse(BaseModel):
    """Response model for concept path."""
    
    source_id: int = Field(..., description="Source concept ID")
    target_id: int = Field(..., description="Target concept ID")
    path_length: int = Field(..., description="Path length")
    concepts: List[GraphConceptResponse] = Field(..., description="Concepts in path")
    relationships: List[GraphRelationshipResponse] = Field(..., description="Relationships in path")


@router.get("/concept/{concept_id}", response_model=GraphConceptResponse, summary="Get concept from graph")
async def get_graph_concept(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> GraphConceptResponse:
    """
    Get a specific concept from the JanusGraph.
    
    Args:
        concept_id: SNOMED-CT concept ID
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Concept vertex data
        
    Raises:
        HTTPException: If concept not found or query fails
    """
    try:
        concept = janusgraph_manager.get_concept_by_id(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found in graph"
            )
        
        return GraphConceptResponse(
            concept_id=concept.get("conceptId", concept_id),
            fully_specified_name=concept.get("fullySpecifiedName", ""),
            active=concept.get("active", True),
            module_id=concept.get("moduleId", 0),
            definition_status_id=concept.get("definitionStatusId", 0),
            created_time=concept.get("createdTime")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concept from graph: {str(e)}"
        )


@router.get("/concept/{concept_id}/relationships", response_model=List[GraphRelationshipResponse], summary="Get concept relationships")
async def get_concept_relationships(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    direction: str = Query("out", regex="^(in|out|both)$", description="Relationship direction"),
    relationship_types: Optional[str] = Query(None, description="Comma-separated relationship type IDs"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of relationships"),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> List[GraphRelationshipResponse]:
    """
    Get relationships for a specific concept from the graph.
    
    Args:
        concept_id: SNOMED-CT concept ID
        direction: Relationship direction (in, out, both)
        relationship_types: Comma-separated list of relationship type IDs
        limit: Maximum number of relationships to return
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        List of relationships
        
    Raises:
        HTTPException: If concept not found or query fails
    """
    try:
        # Parse relationship types if provided
        type_list = None
        if relationship_types:
            try:
                type_list = [int(t.strip()) for t in relationship_types.split(',')]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid relationship type IDs format"
                )
        
        # Get relationships from graph
        relationships = janusgraph_manager.get_concept_relationships(
            concept_id=concept_id,
            direction=direction,
            relationship_types=type_list,
            limit=limit
        )
        
        # Convert to response format
        return [
            GraphRelationshipResponse(
                relationship_id=rel.get("relationshipId", 0),
                source_id=rel.get("sourceId", 0),
                destination_id=rel.get("destinationId", 0),
                type_id=rel.get("typeId", 0),
                relationship_group=rel.get("relationshipGroup", 0),
                active=rel.get("active", True),
                characteristic_type_id=rel.get("characteristicTypeId", 0),
                modifier_id=rel.get("modifierId", 0),
                edge_label=rel.get("edgeLabel")
            )
            for rel in relationships
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relationships from graph: {str(e)}"
        )


@router.get("/concept/{concept_id}/parents", response_model=List[GraphConceptResponse], summary="Get concept parents")
async def get_concept_parents(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    max_depth: int = Query(1, ge=1, le=10, description="Maximum traversal depth"),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> List[GraphConceptResponse]:
    """
    Get hierarchical parents (IS-A relationships) for a concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        max_depth: Maximum depth to traverse
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        List of parent concepts
        
    Raises:
        HTTPException: If concept not found or query fails
    """
    try:
        parents = janusgraph_manager.find_hierarchical_parents(
            concept_id=concept_id,
            max_depth=max_depth
        )
        
        return [
            GraphConceptResponse(
                concept_id=parent.get("conceptId", 0),
                fully_specified_name=parent.get("fullySpecifiedName", ""),
                active=parent.get("active", True),
                module_id=parent.get("moduleId", 0),
                definition_status_id=parent.get("definitionStatusId", 0),
                created_time=parent.get("createdTime")
            )
            for parent in parents
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find parent concepts: {str(e)}"
        )


@router.get("/concept/{concept_id}/children", response_model=List[GraphConceptResponse], summary="Get concept children")
async def get_concept_children(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    max_depth: int = Query(1, ge=1, le=10, description="Maximum traversal depth"),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> List[GraphConceptResponse]:
    """
    Get hierarchical children (IS-A relationships) for a concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        max_depth: Maximum depth to traverse
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        List of child concepts
        
    Raises:
        HTTPException: If concept not found or query fails
    """
    try:
        children = janusgraph_manager.find_hierarchical_children(
            concept_id=concept_id,
            max_depth=max_depth
        )
        
        return [
            GraphConceptResponse(
                concept_id=child.get("conceptId", 0),
                fully_specified_name=child.get("fullySpecifiedName", ""),
                active=child.get("active", True),
                module_id=child.get("moduleId", 0),
                definition_status_id=child.get("definitionStatusId", 0),
                created_time=child.get("createdTime")
            )
            for child in children
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find child concepts: {str(e)}"
        )


@router.post("/traverse", response_model=HierarchyResponse, summary="Traverse concept hierarchy")
async def traverse_hierarchy(
    traversal_request: HierarchyTraversalRequest,
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> HierarchyResponse:
    """
    Traverse concept hierarchy in specified direction.
    
    Args:
        traversal_request: Traversal parameters
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Hierarchy traversal results
        
    Raises:
        HTTPException: If traversal fails
    """
    try:
        concepts = []
        relationships = []
        actual_depth = 0
        
        if traversal_request.direction in ["up", "both"]:
            parents = janusgraph_manager.find_hierarchical_parents(
                concept_id=traversal_request.concept_id,
                max_depth=traversal_request.max_depth
            )
            concepts.extend(parents)
            actual_depth = max(actual_depth, len(parents))
        
        if traversal_request.direction in ["down", "both"]:
            children = janusgraph_manager.find_hierarchical_children(
                concept_id=traversal_request.concept_id,
                max_depth=traversal_request.max_depth
            )
            concepts.extend(children)
            actual_depth = max(actual_depth, len(children))
        
        # Convert concepts to response format
        concept_responses = [
            GraphConceptResponse(
                concept_id=concept.get("conceptId", 0),
                fully_specified_name=concept.get("fullySpecifiedName", ""),
                active=concept.get("active", True),
                module_id=concept.get("moduleId", 0),
                definition_status_id=concept.get("definitionStatusId", 0),
                created_time=concept.get("createdTime")
            )
            for concept in concepts
        ]
        
        return HierarchyResponse(
            root_concept_id=traversal_request.concept_id,
            direction=traversal_request.direction,
            max_depth=traversal_request.max_depth,
            concepts=concept_responses,
            relationships=[],  # TODO: Include relationships in traversal
            depth_reached=actual_depth
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hierarchy traversal failed: {str(e)}"
        )


@router.get("/common-ancestors/{concept_id1}/{concept_id2}", summary="Find common ancestors")
async def find_common_ancestors(
    concept_id1: int = Field(..., description="First concept ID"),
    concept_id2: int = Field(..., description="Second concept ID"),
    max_depth: int = Query(5, ge=1, le=10, description="Maximum search depth"),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> Dict[str, Any]:
    """
    Find common ancestors of two concepts.
    
    Args:
        concept_id1: First SNOMED-CT concept ID
        concept_id2: Second SNOMED-CT concept ID
        max_depth: Maximum depth to search
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Common ancestors and analysis
        
    Raises:
        HTTPException: If concepts not found or query fails
    """
    try:
        # Get parents for both concepts
        parents1 = janusgraph_manager.find_hierarchical_parents(concept_id1, max_depth)
        parents2 = janusgraph_manager.find_hierarchical_parents(concept_id2, max_depth)
        
        # Find common ancestors
        parents1_ids = {p.get("conceptId") for p in parents1}
        parents2_ids = {p.get("conceptId") for p in parents2}
        common_ids = parents1_ids.intersection(parents2_ids)
        
        # Get common ancestor details
        common_ancestors = []
        for parent in parents1:
            if parent.get("conceptId") in common_ids:
                common_ancestors.append(GraphConceptResponse(
                    concept_id=parent.get("conceptId", 0),
                    fully_specified_name=parent.get("fullySpecifiedName", ""),
                    active=parent.get("active", True),
                    module_id=parent.get("moduleId", 0),
                    definition_status_id=parent.get("definitionStatusId", 0),
                    created_time=parent.get("createdTime")
                ))
        
        return {
            "concept_id1": concept_id1,
            "concept_id2": concept_id2,
            "common_ancestors": common_ancestors,
            "total_common_ancestors": len(common_ancestors),
            "search_depth": max_depth
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find common ancestors: {str(e)}"
        )


@router.get("/statistics", summary="Get graph statistics")
async def get_graph_statistics(
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the JanusGraph.
    
    Args:
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Graph statistics and metrics
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        stats = janusgraph_manager.get_graph_statistics()
        
        return {
            "graph_name": janusgraph_manager.graph_name,
            "connection_info": {
                "host": janusgraph_manager.host,
                "port": janusgraph_manager.port,
                "connected": janusgraph_manager.connected
            },
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph statistics: {str(e)}"
        )


@router.get("/relationship-types", summary="Get relationship type information")
async def get_relationship_types(
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> Dict[str, Any]:
    """
    Get information about relationship types used in the graph.
    
    Args:
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Relationship type information
    """
    try:
        # Map relationship types to human-readable names
        relationship_types = {
            116680003: {"name": "Is a", "description": "Core hierarchical relationship"},
            363698007: {"name": "Finding site", "description": "Anatomical location of finding"},
            246075003: {"name": "Causative agent", "description": "Agent causing the condition"},
            123005000: {"name": "Part of", "description": "Structural relationship"},
            363704007: {"name": "Procedure site", "description": "Anatomical location of procedure"},
            263502005: {"name": "Clinical course", "description": "Course of condition"},
            246112005: {"name": "Severity", "description": "Severity of condition"},
            116676008: {"name": "Associated morphology", "description": "Morphological changes"},
            370135005: {"name": "Pathological process", "description": "Underlying pathological process"},
            260686004: {"name": "Method", "description": "Method or technique"}
        }
        
        return {
            "relationship_types": relationship_types,
            "edge_labels": janusgraph_manager.EDGE_LABELS,
            "total_types": len(relationship_types)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relationship types: {str(e)}"
        ) 