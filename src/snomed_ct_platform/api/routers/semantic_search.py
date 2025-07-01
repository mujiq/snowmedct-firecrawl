"""
Semantic search router for Milvus vector similarity queries.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..dependencies import get_milvus_manager, get_embedding_model_manager, get_postgres_manager
from ..config import settings

router = APIRouter()


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    limit: int = Field(default=10, description="Maximum number of results", ge=1, le=100)
    score_threshold: float = Field(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0)
    active_only: bool = Field(default=True, description="Search only active concepts")


class SemanticSearchResult(BaseModel):
    """Response model for semantic search result."""
    
    concept_id: int = Field(..., description="SNOMED-CT concept ID")
    term: str = Field(..., description="Concept term")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    active: bool = Field(..., description="Whether the concept is active")
    module_id: int = Field(..., description="Module ID")
    distance: float = Field(..., description="Vector distance")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""
    
    query: str = Field(..., description="Original search query")
    results: List[SemanticSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results returned")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")


class BatchSemanticSearchRequest(BaseModel):
    """Request model for batch semantic search."""
    
    queries: List[str] = Field(..., description="List of search queries", max_items=50)
    limit: int = Field(default=5, description="Maximum results per query", ge=1, le=50)
    score_threshold: float = Field(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0)
    active_only: bool = Field(default=True, description="Search only active concepts")


class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings."""
    
    text: str = Field(..., description="Text to embed", min_length=1, max_length=1000)


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    text: str = Field(..., description="Original text")
    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


@router.post("/search", response_model=SemanticSearchResponse, summary="Semantic similarity search")
async def semantic_search(
    search_request: SemanticSearchRequest,
    milvus_manager = Depends(get_milvus_manager),
    model_manager = Depends(get_embedding_model_manager)
) -> SemanticSearchResponse:
    """
    Perform semantic similarity search on SNOMED-CT concepts.
    
    Args:
        search_request: Search parameters
        milvus_manager: Milvus database manager
        model_manager: Embedding model manager
        
    Returns:
        Semantic search results with similarity scores
        
    Raises:
        HTTPException: If search fails
    """
    import time
    start_time = time.time()
    
    try:
        # Generate embedding for query text
        query_embedding = model_manager.generate_embedding(search_request.query)
        
        # Prepare search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}  # Adjust based on index type
        }
        
        # Define output fields to retrieve
        output_fields = ["concept_id", "term", "active", "module_id"]
        
        # Perform vector similarity search
        search_results = milvus_manager.search_similar(
            query_embedding=query_embedding,
            limit=search_request.limit,
            search_params=search_params,
            output_fields=output_fields
        )
        
        # Process results and apply filters
        processed_results = []
        for result in search_results:
            # Apply score threshold
            if result.get("score", 0.0) < search_request.score_threshold:
                continue
                
            # Apply active filter
            if search_request.active_only and not result.get("active", True):
                continue
            
            processed_results.append(SemanticSearchResult(
                concept_id=result["concept_id"],
                term=result["term"],
                similarity_score=result.get("score", 0.0),
                active=result.get("active", True),
                module_id=result.get("module_id", 0),
                distance=result.get("distance", 0.0)
            ))
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SemanticSearchResponse(
            query=search_request.query,
            results=processed_results,
            total_results=len(processed_results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post("/batch-search", summary="Batch semantic search")
async def batch_semantic_search(
    search_request: BatchSemanticSearchRequest,
    milvus_manager = Depends(get_milvus_manager),
    model_manager = Depends(get_embedding_model_manager)
) -> Dict[str, Any]:
    """
    Perform batch semantic similarity search on multiple queries.
    
    Args:
        search_request: Batch search parameters
        milvus_manager: Milvus database manager
        model_manager: Embedding model manager
        
    Returns:
        Batch search results for all queries
        
    Raises:
        HTTPException: If batch search fails
    """
    import time
    start_time = time.time()
    
    try:
        # Generate embeddings for all queries
        query_embeddings = model_manager.generate_embeddings_batch(
            search_request.queries,
            show_progress=False
        )
        
        # Perform searches for each query
        all_results = {}
        for i, (query, embedding) in enumerate(zip(search_request.queries, query_embeddings)):
            try:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 16}
                }
                
                output_fields = ["concept_id", "term", "active", "module_id"]
                
                results = milvus_manager.search_similar(
                    query_embedding=embedding,
                    limit=search_request.limit,
                    search_params=search_params,
                    output_fields=output_fields
                )
                
                # Filter results
                filtered_results = []
                for result in results:
                    if result.get("score", 0.0) >= search_request.score_threshold:
                        if not search_request.active_only or result.get("active", True):
                            filtered_results.append({
                                "concept_id": result["concept_id"],
                                "term": result["term"],
                                "similarity_score": result.get("score", 0.0),
                                "active": result.get("active", True),
                                "module_id": result.get("module_id", 0),
                                "distance": result.get("distance", 0.0)
                            })
                
                all_results[query] = filtered_results
                
            except Exception as e:
                all_results[query] = {"error": str(e)}
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "queries": search_request.queries,
            "results": all_results,
            "total_queries": len(search_request.queries),
            "search_time_ms": search_time_ms
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch semantic search failed: {str(e)}"
        )


@router.post("/embed", response_model=EmbeddingResponse, summary="Generate text embedding")
async def generate_embedding(
    embedding_request: EmbeddingRequest,
    model_manager = Depends(get_embedding_model_manager)
) -> EmbeddingResponse:
    """
    Generate embedding vector for input text.
    
    Args:
        embedding_request: Text to embed
        model_manager: Embedding model manager
        
    Returns:
        Generated embedding vector
        
    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        # Generate embedding
        embedding = model_manager.generate_embedding(embedding_request.text)
        
        return EmbeddingResponse(
            text=embedding_request.text,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.get("/similar/{concept_id}", response_model=List[SemanticSearchResult], summary="Find similar concepts")
async def find_similar_concepts(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of similar concepts"),
    score_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    milvus_manager = Depends(get_milvus_manager),
    postgres_manager = Depends(get_postgres_manager)
) -> List[SemanticSearchResult]:
    """
    Find concepts similar to a given concept ID.
    
    Args:
        concept_id: SNOMED-CT concept ID to find similar concepts for
        limit: Maximum number of similar concepts to return
        score_threshold: Minimum similarity score threshold
        milvus_manager: Milvus database manager
        postgres_manager: PostgreSQL database manager
        
    Returns:
        List of similar concepts
        
    Raises:
        HTTPException: If concept not found or search fails
    """
    try:
        # First verify the concept exists in PostgreSQL
        concept = postgres_manager.get_concept_by_id(concept_id)
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
        
        # Get the embedding for this concept from Milvus
        # Note: This requires a method to retrieve embeddings by concept_id
        # For now, we'll use a workaround by searching for the concept itself
        
        # Get concept description for embedding generation
        descriptions = postgres_manager.get_descriptions_by_concept_id(concept_id)
        if not descriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No descriptions found for concept {concept_id}"
            )
        
        # Use the first active description (preferably FSN)
        concept_term = None
        for desc in descriptions:
            if desc['active']:
                concept_term = desc['term']
                if desc['type_id'] == 900000000000003001:  # FSN
                    break
        
        if not concept_term:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active descriptions found for concept {concept_id}"
            )
        
        # Generate embedding for the concept term
        from ..dependencies import get_embedding_model_manager
        model_manager = await get_embedding_model_manager()
        concept_embedding = model_manager.generate_embedding(concept_term)
        
        # Search for similar concepts
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        output_fields = ["concept_id", "term", "active", "module_id"]
        
        results = milvus_manager.search_similar(
            query_embedding=concept_embedding,
            limit=limit + 1,  # +1 to account for the original concept
            search_params=search_params,
            output_fields=output_fields
        )
        
        # Filter out the original concept and apply thresholds
        similar_concepts = []
        for result in results:
            if result["concept_id"] == concept_id:
                continue  # Skip the original concept
            
            if result.get("score", 0.0) >= score_threshold:
                similar_concepts.append(SemanticSearchResult(
                    concept_id=result["concept_id"],
                    term=result["term"],
                    similarity_score=result.get("score", 0.0),
                    active=result.get("active", True),
                    module_id=result.get("module_id", 0),
                    distance=result.get("distance", 0.0)
                ))
        
        return similar_concepts[:limit]  # Ensure we don't exceed the requested limit
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar concepts: {str(e)}"
        )


@router.get("/collection-stats", summary="Get collection statistics")
async def get_collection_stats(
    milvus_manager = Depends(get_milvus_manager)
) -> Dict[str, Any]:
    """
    Get statistics about the Milvus collection.
    
    Args:
        milvus_manager: Milvus database manager
        
    Returns:
        Collection statistics and information
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        stats = milvus_manager.get_collection_stats()
        return {
            "collection_name": milvus_manager.collection_name,
            "statistics": stats,
            "embedding_dimension": milvus_manager.embedding_dim
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection statistics: {str(e)}"
        ) 