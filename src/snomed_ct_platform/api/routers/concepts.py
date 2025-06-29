"""
Concepts router for SNOMED-CT concept queries.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..dependencies import get_postgres_manager
from ..config import settings

router = APIRouter()


class ConceptResponse(BaseModel):
    """Response model for SNOMED-CT concept."""
    
    id: int = Field(..., description="SNOMED-CT concept ID")
    effective_time: str = Field(..., description="Effective time")
    active: bool = Field(..., description="Whether the concept is active")
    module_id: int = Field(..., description="Module ID")
    definition_status_id: int = Field(..., description="Definition status ID")


class ConceptListResponse(BaseModel):
    """Response model for paginated concept list."""
    
    concepts: List[ConceptResponse] = Field(..., description="List of concepts")
    total: int = Field(..., description="Total number of concepts")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class ConceptSearchRequest(BaseModel):
    """Request model for concept search."""
    
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    active_only: bool = Field(default=True, description="Search only active concepts")
    limit: int = Field(default=10, description="Maximum number of results", ge=1, le=100)


@router.get("/", response_model=ConceptListResponse, summary="Get concepts")
async def get_concepts(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=1000, description="Page size"),
    active_only: bool = Query(True, description="Filter to active concepts only"),
    postgres_manager = Depends(get_postgres_manager)
) -> ConceptListResponse:
    """
    Get paginated list of SNOMED-CT concepts.
    
    Args:
        page: Page number (1-based)
        page_size: Number of concepts per page
        active_only: Whether to include only active concepts
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Paginated list of concepts
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        offset = (page - 1) * page_size
        
        # Get concepts from database
        concepts = postgres_manager.get_concepts(
            limit=page_size,
            offset=offset,
            active_only=active_only
        )
        
        # Get total count
        total = postgres_manager.get_concepts_count(active_only=active_only)
        
        # Convert to response format
        concept_responses = [
            ConceptResponse(
                id=concept['id'],
                effective_time=concept['effective_time'].isoformat() if concept['effective_time'] else '',
                active=concept['active'],
                module_id=concept['module_id'],
                definition_status_id=concept['definition_status_id']
            )
            for concept in concepts
        ]
        
        return ConceptListResponse(
            concepts=concept_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concepts: {str(e)}"
        )


@router.get("/{concept_id}", response_model=ConceptResponse, summary="Get concept by ID")
async def get_concept(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> ConceptResponse:
    """
    Get a specific SNOMED-CT concept by ID.
    
    Args:
        concept_id: SNOMED-CT concept ID
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Concept details
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        concept = postgres_manager.get_concept_by_id(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
            
        return ConceptResponse(
            id=concept['id'],
            effective_time=concept['effective_time'].isoformat() if concept['effective_time'] else '',
            active=concept['active'],
            module_id=concept['module_id'],
            definition_status_id=concept['definition_status_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concept: {str(e)}"
        )


@router.post("/search", response_model=List[ConceptResponse], summary="Search concepts")
async def search_concepts(
    search_request: ConceptSearchRequest,
    postgres_manager = Depends(get_postgres_manager)
) -> List[ConceptResponse]:
    """
    Search SNOMED-CT concepts by text query.
    
    Args:
        search_request: Search parameters
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of matching concepts
        
    Raises:
        HTTPException: If search fails
    """
    try:
        # Search concepts by description text
        concepts = postgres_manager.search_concepts_by_text(
            query=search_request.query,
            active_only=search_request.active_only,
            limit=search_request.limit
        )
        
        # Convert to response format
        return [
            ConceptResponse(
                id=concept['id'],
                effective_time=concept['effective_time'].isoformat() if concept['effective_time'] else '',
                active=concept['active'],
                module_id=concept['module_id'],
                definition_status_id=concept['definition_status_id']
            )
            for concept in concepts
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search concepts: {str(e)}"
        )


@router.get("/{concept_id}/descriptions", summary="Get concept descriptions")
async def get_concept_descriptions(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    active_only: bool = Query(True, description="Filter to active descriptions only"),
    postgres_manager = Depends(get_postgres_manager)
) -> List[Dict[str, Any]]:
    """
    Get all descriptions for a specific concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        active_only: Whether to include only active descriptions
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of concept descriptions
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        descriptions = postgres_manager.get_descriptions_by_concept_id(
            concept_id=concept_id,
            active_only=active_only
        )
        
        return descriptions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concept descriptions: {str(e)}"
        )


@router.get("/{concept_id}/relationships", summary="Get concept relationships")
async def get_concept_relationships(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    active_only: bool = Query(True, description="Filter to active relationships only"),
    direction: str = Query("both", regex="^(source|destination|both)$", description="Relationship direction"),
    postgres_manager = Depends(get_postgres_manager)
) -> List[Dict[str, Any]]:
    """
    Get all relationships for a specific concept.
    
    Args:
        concept_id: SNOMED-CT concept ID  
        active_only: Whether to include only active relationships
        direction: Relationship direction filter (source, destination, both)
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of concept relationships
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        relationships = postgres_manager.get_relationships_by_concept_id(
            concept_id=concept_id,
            active_only=active_only,
            direction=direction
        )
        
        return relationships
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve concept relationships: {str(e)}"
        ) 