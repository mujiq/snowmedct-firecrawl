"""
Relationships router for SNOMED-CT relationship queries.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..dependencies import get_postgres_manager
from ..config import settings

router = APIRouter()


class RelationshipResponse(BaseModel):
    """Response model for SNOMED-CT relationship."""
    
    id: int = Field(..., description="SNOMED-CT relationship ID")
    effective_time: str = Field(..., description="Effective time")
    active: bool = Field(..., description="Whether the relationship is active")
    module_id: int = Field(..., description="Module ID")
    source_id: int = Field(..., description="Source concept ID")
    destination_id: int = Field(..., description="Destination concept ID")
    relationship_group: int = Field(..., description="Relationship group number")
    type_id: int = Field(..., description="Relationship type ID")
    characteristic_type_id: int = Field(..., description="Characteristic type ID")
    modifier_id: int = Field(..., description="Modifier ID")


class RelationshipListResponse(BaseModel):
    """Response model for paginated relationship list."""
    
    relationships: List[RelationshipResponse] = Field(..., description="List of relationships")
    total: int = Field(..., description="Total number of relationships")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class RelationshipSearchRequest(BaseModel):
    """Request model for relationship search."""
    
    source_id: Optional[int] = Field(None, description="Filter by source concept ID")
    destination_id: Optional[int] = Field(None, description="Filter by destination concept ID")
    type_id: Optional[int] = Field(None, description="Filter by relationship type ID")
    active_only: bool = Field(default=True, description="Search only active relationships")
    limit: int = Field(default=20, description="Maximum number of results", ge=1, le=1000)


@router.get("/", response_model=RelationshipListResponse, summary="Get relationships")
async def get_relationships(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=1000, description="Page size"),
    active_only: bool = Query(True, description="Filter to active relationships only"),
    type_id: Optional[int] = Query(None, description="Filter by relationship type ID"),
    source_id: Optional[int] = Query(None, description="Filter by source concept ID"),
    destination_id: Optional[int] = Query(None, description="Filter by destination concept ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> RelationshipListResponse:
    """
    Get paginated list of SNOMED-CT relationships.
    
    Args:
        page: Page number (1-based)
        page_size: Number of relationships per page
        active_only: Whether to include only active relationships
        type_id: Optional relationship type ID to filter by
        source_id: Optional source concept ID to filter by
        destination_id: Optional destination concept ID to filter by
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Paginated list of relationships
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        offset = (page - 1) * page_size
        
        # Handle specific concept filtering
        if source_id or destination_id:
            concept_id = source_id or destination_id
            direction = "source" if source_id else "destination"
            
            relationships = postgres_manager.get_relationships_by_concept_id(
                concept_id=concept_id,
                active_only=active_only,
                direction=direction
            )
            
            # Filter by type_id if specified
            if type_id:
                relationships = [r for r in relationships if r['type_id'] == type_id]
            
            # Apply pagination manually
            total = len(relationships)
            relationships = relationships[offset:offset + page_size]
        else:
            # Get all relationships with pagination
            relationships = postgres_manager.get_relationships(
                limit=page_size,
                offset=offset,
                active_only=active_only,
                type_id=type_id
            )
            total = postgres_manager.get_relationships_count(
                active_only=active_only,
                type_id=type_id
            )
        
        # Convert to response format
        relationship_responses = [
            RelationshipResponse(
                id=rel['id'],
                effective_time=rel['effective_time'],
                active=rel['active'],
                module_id=rel['module_id'],
                source_id=rel['source_id'],
                destination_id=rel['destination_id'],
                relationship_group=rel['relationship_group'],
                type_id=rel['type_id'],
                characteristic_type_id=rel['characteristic_type_id'],
                modifier_id=rel['modifier_id']
            )
            for rel in relationships
        ]
        
        return RelationshipListResponse(
            relationships=relationship_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relationships: {str(e)}"
        )


@router.get("/{relationship_id}", response_model=RelationshipResponse, summary="Get relationship by ID")
async def get_relationship(
    relationship_id: int = Field(..., description="SNOMED-CT relationship ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> RelationshipResponse:
    """
    Get a specific SNOMED-CT relationship by ID.
    
    Args:
        relationship_id: SNOMED-CT relationship ID
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Relationship details
        
    Raises:
        HTTPException: If relationship not found or database query fails
    """
    try:
        # Query for specific relationship
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            query = """
                SELECT id, effective_time, active, module_id, source_id, destination_id,
                       relationship_group, type_id, characteristic_type_id, modifier_id
                FROM relationships
                WHERE id = :relationship_id
            """
            
            result = conn.execute(text(query), {"relationship_id": relationship_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Relationship {relationship_id} not found"
                )
            
            relationship = dict(row._mapping)
            
            return RelationshipResponse(
                id=relationship['id'],
                effective_time=relationship['effective_time'],
                active=relationship['active'],
                module_id=relationship['module_id'],
                source_id=relationship['source_id'],
                destination_id=relationship['destination_id'],
                relationship_group=relationship['relationship_group'],
                type_id=relationship['type_id'],
                characteristic_type_id=relationship['characteristic_type_id'],
                modifier_id=relationship['modifier_id']
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relationship: {str(e)}"
        )


@router.post("/search", response_model=List[RelationshipResponse], summary="Search relationships")
async def search_relationships(
    search_request: RelationshipSearchRequest,
    postgres_manager = Depends(get_postgres_manager)
) -> List[RelationshipResponse]:
    """
    Search SNOMED-CT relationships by various criteria.
    
    Args:
        search_request: Search parameters
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of matching relationships
        
    Raises:
        HTTPException: If search fails
    """
    try:
        # Build custom query based on search criteria
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            
            query_parts = []
            params = {}
            
            base_query = """
                SELECT id, effective_time, active, module_id, source_id, destination_id,
                       relationship_group, type_id, characteristic_type_id, modifier_id
                FROM relationships
                WHERE 1=1
            """
            
            if search_request.active_only:
                query_parts.append("AND active = true")
            
            if search_request.source_id:
                query_parts.append("AND source_id = :source_id")
                params["source_id"] = search_request.source_id
            
            if search_request.destination_id:
                query_parts.append("AND destination_id = :destination_id")
                params["destination_id"] = search_request.destination_id
            
            if search_request.type_id:
                query_parts.append("AND type_id = :type_id")
                params["type_id"] = search_request.type_id
            
            full_query = base_query + " " + " ".join(query_parts)
            full_query += f" ORDER BY source_id, type_id, id LIMIT {search_request.limit}"
            
            result = conn.execute(text(full_query), params)
            relationships = [dict(row._mapping) for row in result]
        
        # Convert to response format
        return [
            RelationshipResponse(
                id=rel['id'],
                effective_time=rel['effective_time'],
                active=rel['active'],
                module_id=rel['module_id'],
                source_id=rel['source_id'],
                destination_id=rel['destination_id'],
                relationship_group=rel['relationship_group'],
                type_id=rel['type_id'],
                characteristic_type_id=rel['characteristic_type_id'],
                modifier_id=rel['modifier_id']
            )
            for rel in relationships
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search relationships: {str(e)}"
        )


@router.get("/types", summary="Get relationship types")
async def get_relationship_types(
    postgres_manager = Depends(get_postgres_manager)
) -> Dict[str, Any]:
    """
    Get available relationship types from the database.
    
    Args:
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Dictionary of relationship types and their counts
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            query = """
                SELECT type_id, COUNT(*) as count
                FROM relationships
                WHERE active = true
                GROUP BY type_id
                ORDER BY count DESC
            """
            
            result = conn.execute(text(query))
            types = [dict(row._mapping) for row in result]
            
            # Map common SNOMED-CT relationship type IDs to human-readable names
            type_names = {
                116680003: "Is a",
                363698007: "Finding site",
                246075003: "Causative agent",
                405815000: "Procedure device",
                424226004: "Using device",
                47429007: "Associated with",
                260686004: "Method",
                370129005: "Measurement method",
                246093002: "Component",
                246501002: "Technique"
            }
            
            return {
                "relationship_types": [
                    {
                        "type_id": t["type_id"],
                        "name": type_names.get(t["type_id"], f"Type {t['type_id']}"),
                        "count": t["count"]
                    }
                    for t in types
                ],
                "total_types": len(types)
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relationship types: {str(e)}"
        )


@router.get("/by-concept/{concept_id}", response_model=List[RelationshipResponse], summary="Get relationships by concept")
async def get_relationships_by_concept(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    direction: str = Query("both", regex="^(source|destination|both)$", description="Relationship direction"),
    active_only: bool = Query(True, description="Filter to active relationships only"),
    type_id: Optional[int] = Query(None, description="Filter by relationship type ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> List[RelationshipResponse]:
    """
    Get all relationships for a specific concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        direction: Relationship direction filter (source, destination, both)
        active_only: Whether to include only active relationships
        type_id: Optional relationship type ID to filter by
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of relationships for the concept
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        relationships = postgres_manager.get_relationships_by_concept_id(
            concept_id=concept_id,
            active_only=active_only,
            direction=direction
        )
        
        # Filter by type_id if specified
        if type_id:
            relationships = [r for r in relationships if r['type_id'] == type_id]
        
        # Convert to response format
        return [
            RelationshipResponse(
                id=rel['id'],
                effective_time=rel['effective_time'],
                active=rel['active'],
                module_id=rel['module_id'],
                source_id=rel['source_id'],
                destination_id=rel['destination_id'],
                relationship_group=rel['relationship_group'],
                type_id=rel['type_id'],
                characteristic_type_id=rel['characteristic_type_id'],
                modifier_id=rel['modifier_id']
            )
            for rel in relationships
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve relationships for concept {concept_id}: {str(e)}"
        )


@router.get("/hierarchy/{concept_id}", summary="Get concept hierarchy")
async def get_concept_hierarchy(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    depth: int = Query(1, ge=1, le=5, description="Hierarchy depth to traverse"),
    postgres_manager = Depends(get_postgres_manager)
) -> Dict[str, Any]:
    """
    Get hierarchical relationships (IS_A) for a concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        depth: Number of hierarchy levels to traverse
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Hierarchical structure with parents and children
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            
            # Get parents (IS_A relationships where this concept is the source)
            parent_query = """
                SELECT destination_id as parent_id
                FROM relationships
                WHERE source_id = :concept_id
                AND type_id = 116680003  -- IS_A relationship type
                AND active = true
            """
            
            parent_result = conn.execute(text(parent_query), {"concept_id": concept_id})
            parents = [row.parent_id for row in parent_result]
            
            # Get children (IS_A relationships where this concept is the destination)
            child_query = """
                SELECT source_id as child_id
                FROM relationships
                WHERE destination_id = :concept_id
                AND type_id = 116680003  -- IS_A relationship type
                AND active = true
            """
            
            child_result = conn.execute(text(child_query), {"concept_id": concept_id})
            children = [row.child_id for row in child_result]
            
            return {
                "concept_id": concept_id,
                "parents": parents,
                "children": children,
                "parent_count": len(parents),
                "child_count": len(children),
                "depth_requested": depth,
                "note": "Multi-level hierarchy traversal requires recursive queries - implement as needed"
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get hierarchy for concept {concept_id}: {str(e)}"
        ) 