"""
Descriptions router for SNOMED-CT description queries.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..dependencies import get_postgres_manager
from ..config import settings

router = APIRouter()


class DescriptionResponse(BaseModel):
    """Response model for SNOMED-CT description."""
    
    id: int = Field(..., description="SNOMED-CT description ID")
    effective_time: str = Field(..., description="Effective time")
    active: bool = Field(..., description="Whether the description is active")
    module_id: int = Field(..., description="Module ID")
    concept_id: int = Field(..., description="Related concept ID")
    language_code: str = Field(..., description="Language code (e.g., 'en')")
    type_id: int = Field(..., description="Description type ID")
    term: str = Field(..., description="Description text")
    case_significance_id: int = Field(..., description="Case significance ID")


class DescriptionListResponse(BaseModel):
    """Response model for paginated description list."""
    
    descriptions: List[DescriptionResponse] = Field(..., description="List of descriptions")
    total: int = Field(..., description="Total number of descriptions")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class DescriptionSearchRequest(BaseModel):
    """Request model for description search."""
    
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    active_only: bool = Field(default=True, description="Search only active descriptions")
    limit: int = Field(default=10, description="Maximum number of results", ge=1, le=100)


@router.get("/", response_model=DescriptionListResponse, summary="Get descriptions")
async def get_descriptions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=1000, description="Page size"),
    active_only: bool = Query(True, description="Filter to active descriptions only"),
    concept_id: Optional[int] = Query(None, description="Filter by concept ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> DescriptionListResponse:
    """
    Get paginated list of SNOMED-CT descriptions.
    
    Args:
        page: Page number (1-based)
        page_size: Number of descriptions per page
        active_only: Whether to include only active descriptions
        concept_id: Optional concept ID to filter by
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Paginated list of descriptions
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        offset = (page - 1) * page_size
        
        if concept_id:
            # Get descriptions for specific concept
            descriptions = postgres_manager.get_descriptions_by_concept_id(
                concept_id=concept_id,
                active_only=active_only
            )
            # Apply pagination manually for concept-specific queries
            total = len(descriptions)
            descriptions = descriptions[offset:offset + page_size]
        else:
            # Get all descriptions with pagination
            # Note: This would need a new method in postgres_manager for general description listing
            # For now, use search with empty query as fallback
            descriptions = postgres_manager.search_descriptions(
                query="",
                active_only=active_only,
                limit=page_size,
                offset=offset
            )
            total = postgres_manager.get_descriptions_count(active_only=active_only)
        
        # Convert to response format
        description_responses = [
            DescriptionResponse(
                id=desc['id'],
                effective_time=desc['effective_time'],
                active=desc['active'],
                module_id=desc['module_id'],
                concept_id=desc['concept_id'],
                language_code=desc['language_code'],
                type_id=desc['type_id'],
                term=desc['term'],
                case_significance_id=desc['case_significance_id']
            )
            for desc in descriptions
        ]
        
        return DescriptionListResponse(
            descriptions=description_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve descriptions: {str(e)}"
        )


@router.get("/{description_id}", response_model=DescriptionResponse, summary="Get description by ID")
async def get_description(
    description_id: int = Field(..., description="SNOMED-CT description ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> DescriptionResponse:
    """
    Get a specific SNOMED-CT description by ID.
    
    Args:
        description_id: SNOMED-CT description ID
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Description details
        
    Raises:
        HTTPException: If description not found or database query fails
    """
    try:
        # Query for specific description
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            query = """
                SELECT id, effective_time, active, module_id, concept_id,
                       language_code, type_id, term, case_significance_id
                FROM descriptions
                WHERE id = :description_id
            """
            
            result = conn.execute(text(query), {"description_id": description_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Description {description_id} not found"
                )
            
            description = dict(row._mapping)
            
            return DescriptionResponse(
                id=description['id'],
                effective_time=description['effective_time'],
                active=description['active'],
                module_id=description['module_id'],
                concept_id=description['concept_id'],
                language_code=description['language_code'],
                type_id=description['type_id'],
                term=description['term'],
                case_significance_id=description['case_significance_id']
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve description: {str(e)}"
        )


@router.post("/search", response_model=List[DescriptionResponse], summary="Search descriptions")
async def search_descriptions(
    search_request: DescriptionSearchRequest,
    postgres_manager = Depends(get_postgres_manager)
) -> List[DescriptionResponse]:
    """
    Search SNOMED-CT descriptions by text query.
    
    Args:
        search_request: Search parameters
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of matching descriptions
        
    Raises:
        HTTPException: If search fails
    """
    try:
        # Search descriptions by text
        descriptions = postgres_manager.search_descriptions(
            query=search_request.query,
            active_only=search_request.active_only,
            limit=search_request.limit
        )
        
        # Convert to response format
        return [
            DescriptionResponse(
                id=desc['id'],
                effective_time=desc['effective_time'],
                active=desc['active'],
                module_id=desc['module_id'],
                concept_id=desc['concept_id'],
                language_code=desc['language_code'],
                type_id=desc['type_id'],
                term=desc['term'],
                case_significance_id=desc['case_significance_id']
            )
            for desc in descriptions
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search descriptions: {str(e)}"
        )


@router.get("/types", summary="Get description types")
async def get_description_types(
    postgres_manager = Depends(get_postgres_manager)
) -> Dict[str, Any]:
    """
    Get available description types from the database.
    
    Args:
        postgres_manager: PostgreSQL database manager
        
    Returns:
        Dictionary of description types and their counts
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        with postgres_manager.engine.connect() as conn:
            from sqlalchemy import text
            query = """
                SELECT type_id, COUNT(*) as count
                FROM descriptions
                WHERE active = true
                GROUP BY type_id
                ORDER BY count DESC
            """
            
            result = conn.execute(text(query))
            types = [dict(row._mapping) for row in result]
            
            # Map common SNOMED-CT description type IDs to human-readable names
            type_names = {
                900000000000003001: "Fully Specified Name",
                900000000000013009: "Synonym",
                900000000000550004: "Definition"
            }
            
            return {
                "description_types": [
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
            detail=f"Failed to get description types: {str(e)}"
        )


@router.get("/by-concept/{concept_id}", response_model=List[DescriptionResponse], summary="Get descriptions by concept")
async def get_descriptions_by_concept(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    active_only: bool = Query(True, description="Filter to active descriptions only"),
    type_id: Optional[int] = Query(None, description="Filter by description type ID"),
    postgres_manager = Depends(get_postgres_manager)
) -> List[DescriptionResponse]:
    """
    Get all descriptions for a specific concept.
    
    Args:
        concept_id: SNOMED-CT concept ID
        active_only: Whether to include only active descriptions
        type_id: Optional description type ID to filter by
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of descriptions for the concept
        
    Raises:
        HTTPException: If concept not found or database query fails
    """
    try:
        descriptions = postgres_manager.get_descriptions_by_concept_id(
            concept_id=concept_id,
            active_only=active_only
        )
        
        # Filter by type_id if specified
        if type_id:
            descriptions = [d for d in descriptions if d['type_id'] == type_id]
        
        # Convert to response format
        return [
            DescriptionResponse(
                id=desc['id'],
                effective_time=desc['effective_time'],
                active=desc['active'],
                module_id=desc['module_id'],
                concept_id=desc['concept_id'],
                language_code=desc['language_code'],
                type_id=desc['type_id'],
                term=desc['term'],
                case_significance_id=desc['case_significance_id']
            )
            for desc in descriptions
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve descriptions for concept {concept_id}: {str(e)}"
        ) 