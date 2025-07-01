"""
Unified search router for multi-modal queries across all databases.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
import asyncio
import time

from ..dependencies import (
    get_postgres_manager, 
    get_milvus_manager, 
    get_janusgraph_manager,
    get_embedding_model_manager
)
from ..config import settings

router = APIRouter()


class UnifiedSearchRequest(BaseModel):
    """Request model for unified multi-modal search."""
    
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    include_textual: bool = Field(default=True, description="Include textual search results")
    include_semantic: bool = Field(default=True, description="Include semantic similarity results")
    include_graph: bool = Field(default=True, description="Include graph-based results")
    limit_per_source: int = Field(default=10, description="Maximum results per source", ge=1, le=50)
    score_threshold: float = Field(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0)
    active_only: bool = Field(default=True, description="Search only active concepts")


class UnifiedConceptResult(BaseModel):
    """Unified concept result combining data from all sources."""
    
    concept_id: int = Field(..., description="SNOMED-CT concept ID")
    fully_specified_name: str = Field(..., description="Fully specified name")
    synonyms: List[str] = Field(default=[], description="Synonym terms")
    active: bool = Field(..., description="Whether the concept is active")
    module_id: int = Field(..., description="Module ID")
    
    # Textual search info
    textual_match_score: Optional[float] = Field(None, description="Textual match relevance")
    matched_terms: List[str] = Field(default=[], description="Matched description terms")
    
    # Semantic search info
    semantic_similarity: Optional[float] = Field(None, description="Semantic similarity score")
    embedding_distance: Optional[float] = Field(None, description="Vector distance")
    
    # Graph info
    hierarchical_level: Optional[int] = Field(None, description="Level in hierarchy")
    parent_concepts: List[int] = Field(default=[], description="Direct parent concept IDs")
    child_count: Optional[int] = Field(None, description="Number of child concepts")
    relationship_count: Optional[int] = Field(None, description="Total relationship count")
    
    # Meta info
    source_databases: List[str] = Field(default=[], description="Databases containing this concept")
    overall_relevance: float = Field(..., description="Combined relevance score")


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search."""
    
    query: str = Field(..., description="Original search query")
    results: List[UnifiedConceptResult] = Field(..., description="Unified search results")
    result_counts: Dict[str, int] = Field(..., description="Results count per database")
    search_time_ms: float = Field(..., description="Total search time")
    databases_searched: List[str] = Field(..., description="Databases included in search")


class ConceptEnrichmentRequest(BaseModel):
    """Request model for concept enrichment."""
    
    concept_id: int = Field(..., description="SNOMED-CT concept ID to enrich")
    include_descriptions: bool = Field(default=True, description="Include all descriptions")
    include_relationships: bool = Field(default=True, description="Include relationships")
    include_hierarchy: bool = Field(default=True, description="Include hierarchical context")
    include_similar: bool = Field(default=True, description="Include semantically similar concepts")
    hierarchy_depth: int = Field(default=2, description="Depth for hierarchy traversal", ge=1, le=5)
    similar_limit: int = Field(default=5, description="Number of similar concepts", ge=1, le=20)


class EnrichedConceptResponse(BaseModel):
    """Response model for enriched concept data."""
    
    concept: UnifiedConceptResult = Field(..., description="Core concept information")
    descriptions: List[Dict[str, Any]] = Field(default=[], description="All descriptions")
    relationships: List[Dict[str, Any]] = Field(default=[], description="All relationships")
    parents: List[UnifiedConceptResult] = Field(default=[], description="Parent concepts")
    children: List[UnifiedConceptResult] = Field(default=[], description="Child concepts")
    similar_concepts: List[UnifiedConceptResult] = Field(default=[], description="Semantically similar")
    graph_statistics: Dict[str, Any] = Field(default={}, description="Graph connectivity info")


@router.post("/search", response_model=UnifiedSearchResponse, summary="Unified multi-modal search")
async def unified_search(
    search_request: UnifiedSearchRequest,
    postgres_manager = Depends(get_postgres_manager),
    milvus_manager = Depends(get_milvus_manager),
    janusgraph_manager = Depends(get_janusgraph_manager),
    model_manager = Depends(get_embedding_model_manager)
) -> UnifiedSearchResponse:
    """
    Perform unified search across PostgreSQL, Milvus, and JanusGraph.
    
    Args:
        search_request: Search parameters
        postgres_manager: PostgreSQL database manager
        milvus_manager: Milvus database manager
        janusgraph_manager: JanusGraph database manager
        model_manager: Embedding model manager
        
    Returns:
        Unified search results from all databases
        
    Raises:
        HTTPException: If search fails
    """
    start_time = time.time()
    
    try:
        # Initialize result containers
        textual_results = []
        semantic_results = []
        graph_results = []
        databases_searched = []
        result_counts = {"postgres": 0, "milvus": 0, "janusgraph": 0}
        
        # Perform searches in parallel where possible
        search_tasks = []
        
        # 1. Textual search in PostgreSQL
        if search_request.include_textual:
            try:
                concepts = postgres_manager.search_concepts_by_text(
                    query=search_request.query,
                    active_only=search_request.active_only,
                    limit=search_request.limit_per_source
                )
                textual_results = concepts
                result_counts["postgres"] = len(concepts)
                databases_searched.append("postgres")
            except Exception as e:
                logger.warning(f"PostgreSQL search failed: {e}")
        
        # 2. Semantic search in Milvus
        if search_request.include_semantic:
            try:
                # Generate embedding for query
                query_embedding = model_manager.generate_embedding(search_request.query)
                
                # Perform semantic search
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
                output_fields = ["concept_id", "term", "active", "module_id"]
                
                results = milvus_manager.search_similar(
                    query_embedding=query_embedding,
                    limit=search_request.limit_per_source,
                    search_params=search_params,
                    output_fields=output_fields
                )
                
                # Filter by score threshold
                semantic_results = [
                    r for r in results 
                    if r.get("score", 0.0) >= search_request.score_threshold
                ]
                result_counts["milvus"] = len(semantic_results)
                databases_searched.append("milvus")
            except Exception as e:
                logger.warning(f"Milvus search failed: {e}")
        
        # 3. Graph-based concept discovery
        if search_request.include_graph and textual_results:
            try:
                # Use textual results to find related concepts in graph
                graph_concept_ids = set()
                for concept in textual_results[:5]:  # Use top 5 textual matches
                    concept_id = concept.get("id") or concept.get("concept_id")
                    if concept_id:
                        # Get related concepts from graph
                        try:
                            parents = janusgraph_manager.find_hierarchical_parents(concept_id, 1)
                            children = janusgraph_manager.find_hierarchical_children(concept_id, 1)
                            
                            for parent in parents:
                                graph_concept_ids.add(parent.get("conceptId"))
                            for child in children:
                                graph_concept_ids.add(child.get("conceptId"))
                        except Exception:
                            continue
                
                # Get details for graph concepts from PostgreSQL
                graph_results = []
                for concept_id in list(graph_concept_ids)[:search_request.limit_per_source]:
                    concept = postgres_manager.get_concept_by_id(concept_id)
                    if concept:
                        graph_results.append(concept)
                
                result_counts["janusgraph"] = len(graph_results)
                databases_searched.append("janusgraph")
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
        
        # 4. Combine and rank results
        unified_results = []
        processed_concept_ids = set()
        
        # Process textual results
        for concept in textual_results:
            concept_id = concept.get("id") or concept.get("concept_id")
            if concept_id not in processed_concept_ids:
                processed_concept_ids.add(concept_id)
                
                # Get additional data
                descriptions = postgres_manager.get_descriptions_by_concept_id(concept_id, active_only=True)
                synonyms = [d["term"] for d in descriptions if d["type_id"] == 900000000000013009]  # Synonyms
                matched_terms = [concept.get("matched_term", "")]
                
                unified_result = UnifiedConceptResult(
                    concept_id=concept_id,
                    fully_specified_name=concept.get("matched_term", "") or next(
                        (d["term"] for d in descriptions if d["type_id"] == 900000000000003001), ""
                    ),
                    synonyms=synonyms,
                    active=concept.get("active", True),
                    module_id=concept.get("module_id", 0),
                    textual_match_score=1.0,  # High score for direct text match
                    matched_terms=matched_terms,
                    source_databases=["postgres"],
                    overall_relevance=1.0
                )
                unified_results.append(unified_result)
        
        # Process semantic results
        for result in semantic_results:
            concept_id = result["concept_id"]
            if concept_id not in processed_concept_ids:
                processed_concept_ids.add(concept_id)
                
                # Get additional data from PostgreSQL
                concept = postgres_manager.get_concept_by_id(concept_id)
                descriptions = postgres_manager.get_descriptions_by_concept_id(concept_id, active_only=True)
                synonyms = [d["term"] for d in descriptions if d["type_id"] == 900000000000013009]
                
                unified_result = UnifiedConceptResult(
                    concept_id=concept_id,
                    fully_specified_name=result["term"],
                    synonyms=synonyms,
                    active=result.get("active", True),
                    module_id=result.get("module_id", 0),
                    semantic_similarity=result.get("score", 0.0),
                    embedding_distance=result.get("distance", 0.0),
                    source_databases=["milvus", "postgres"],
                    overall_relevance=result.get("score", 0.0) * 0.8  # Weight semantic matches slightly lower
                )
                unified_results.append(unified_result)
        
        # Process graph results
        for concept in graph_results:
            concept_id = concept.get("id") or concept.get("concept_id")
            if concept_id not in processed_concept_ids:
                processed_concept_ids.add(concept_id)
                
                descriptions = postgres_manager.get_descriptions_by_concept_id(concept_id, active_only=True)
                synonyms = [d["term"] for d in descriptions if d["type_id"] == 900000000000013009]
                fsn = next((d["term"] for d in descriptions if d["type_id"] == 900000000000003001), "")
                
                unified_result = UnifiedConceptResult(
                    concept_id=concept_id,
                    fully_specified_name=fsn,
                    synonyms=synonyms,
                    active=concept.get("active", True),
                    module_id=concept.get("module_id", 0),
                    hierarchical_level=1,  # Related through hierarchy
                    source_databases=["janusgraph", "postgres"],
                    overall_relevance=0.6  # Lower relevance for hierarchically related
                )
                unified_results.append(unified_result)
        
        # Sort by overall relevance
        unified_results.sort(key=lambda x: x.overall_relevance, reverse=True)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return UnifiedSearchResponse(
            query=search_request.query,
            results=unified_results,
            result_counts=result_counts,
            search_time_ms=search_time_ms,
            databases_searched=databases_searched
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unified search failed: {str(e)}"
        )


@router.post("/enrich/{concept_id}", response_model=EnrichedConceptResponse, summary="Enrich concept with multi-modal data")
async def enrich_concept(
    concept_id: int = Field(..., description="SNOMED-CT concept ID"),
    enrichment_request: ConceptEnrichmentRequest = None,
    postgres_manager = Depends(get_postgres_manager),
    milvus_manager = Depends(get_milvus_manager),
    janusgraph_manager = Depends(get_janusgraph_manager),
    model_manager = Depends(get_embedding_model_manager)
) -> EnrichedConceptResponse:
    """
    Enrich a concept with comprehensive data from all databases.
    
    Args:
        concept_id: SNOMED-CT concept ID to enrich
        enrichment_request: Enrichment parameters
        postgres_manager: PostgreSQL database manager
        milvus_manager: Milvus database manager
        janusgraph_manager: JanusGraph database manager
        model_manager: Embedding model manager
        
    Returns:
        Comprehensive concept enrichment data
        
    Raises:
        HTTPException: If concept not found or enrichment fails
    """
    if enrichment_request is None:
        enrichment_request = ConceptEnrichmentRequest(concept_id=concept_id)
    
    try:
        # 1. Get base concept from PostgreSQL
        concept = postgres_manager.get_concept_by_id(concept_id)
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept {concept_id} not found"
            )
        
        # 2. Get descriptions
        descriptions = []
        if enrichment_request.include_descriptions:
            descriptions = postgres_manager.get_descriptions_by_concept_id(concept_id, active_only=True)
        
        # 3. Get relationships
        relationships = []
        if enrichment_request.include_relationships:
            relationships = postgres_manager.get_relationships_by_concept_id(concept_id, active_only=True)
        
        # 4. Get hierarchy information
        parents = []
        children = []
        if enrichment_request.include_hierarchy:
            try:
                parent_data = janusgraph_manager.find_hierarchical_parents(
                    concept_id, enrichment_request.hierarchy_depth
                )
                child_data = janusgraph_manager.find_hierarchical_children(
                    concept_id, enrichment_request.hierarchy_depth
                )
                
                # Convert to unified format
                for parent in parent_data:
                    parents.append(UnifiedConceptResult(
                        concept_id=parent.get("conceptId", 0),
                        fully_specified_name=parent.get("fullySpecifiedName", ""),
                        active=parent.get("active", True),
                        module_id=parent.get("moduleId", 0),
                        source_databases=["janusgraph"],
                        overall_relevance=1.0
                    ))
                
                for child in child_data:
                    children.append(UnifiedConceptResult(
                        concept_id=child.get("conceptId", 0),
                        fully_specified_name=child.get("fullySpecifiedName", ""),
                        active=child.get("active", True),
                        module_id=child.get("moduleId", 0),
                        source_databases=["janusgraph"],
                        overall_relevance=1.0
                    ))
            except Exception as e:
                logger.warning(f"Hierarchy enrichment failed: {e}")
        
        # 5. Get similar concepts
        similar_concepts = []
        if enrichment_request.include_similar:
            try:
                # Get concept's FSN for embedding
                fsn = next((d["term"] for d in descriptions if d["type_id"] == 900000000000003001), "")
                if fsn:
                    concept_embedding = model_manager.generate_embedding(fsn)
                    
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
                    output_fields = ["concept_id", "term", "active", "module_id"]
                    
                    similar_results = milvus_manager.search_similar(
                        query_embedding=concept_embedding,
                        limit=enrichment_request.similar_limit + 1,  # +1 to account for self
                        search_params=search_params,
                        output_fields=output_fields
                    )
                    
                    for result in similar_results:
                        if result["concept_id"] != concept_id:  # Exclude self
                            similar_concepts.append(UnifiedConceptResult(
                                concept_id=result["concept_id"],
                                fully_specified_name=result["term"],
                                active=result.get("active", True),
                                module_id=result.get("module_id", 0),
                                semantic_similarity=result.get("score", 0.0),
                                source_databases=["milvus"],
                                overall_relevance=result.get("score", 0.0)
                            ))
            except Exception as e:
                logger.warning(f"Similar concepts enrichment failed: {e}")
        
        # 6. Create core concept result
        synonyms = [d["term"] for d in descriptions if d["type_id"] == 900000000000013009]
        fsn = next((d["term"] for d in descriptions if d["type_id"] == 900000000000003001), "")
        
        core_concept = UnifiedConceptResult(
            concept_id=concept_id,
            fully_specified_name=fsn,
            synonyms=synonyms,
            active=concept.get("active", True),
            module_id=concept.get("module_id", 0),
            parent_concepts=[p.concept_id for p in parents],
            child_count=len(children),
            relationship_count=len(relationships),
            source_databases=["postgres", "janusgraph", "milvus"],
            overall_relevance=1.0
        )
        
        # 7. Graph statistics
        graph_stats = {
            "parent_count": len(parents),
            "child_count": len(children),
            "relationship_count": len(relationships),
            "similar_concept_count": len(similar_concepts)
        }
        
        return EnrichedConceptResponse(
            concept=core_concept,
            descriptions=descriptions,
            relationships=relationships,
            parents=parents,
            children=children,
            similar_concepts=similar_concepts,
            graph_statistics=graph_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Concept enrichment failed: {str(e)}"
        )


@router.get("/compare/{concept_id1}/{concept_id2}", summary="Compare two concepts")
async def compare_concepts(
    concept_id1: int = Field(..., description="First concept ID"),
    concept_id2: int = Field(..., description="Second concept ID"),
    postgres_manager = Depends(get_postgres_manager),
    milvus_manager = Depends(get_milvus_manager),
    janusgraph_manager = Depends(get_janusgraph_manager),
    model_manager = Depends(get_embedding_model_manager)
) -> Dict[str, Any]:
    """
    Compare two concepts across all dimensions.
    
    Args:
        concept_id1: First SNOMED-CT concept ID
        concept_id2: Second SNOMED-CT concept ID
        postgres_manager: PostgreSQL database manager
        milvus_manager: Milvus database manager
        janusgraph_manager: JanusGraph database manager
        model_manager: Embedding model manager
        
    Returns:
        Comprehensive comparison analysis
        
    Raises:
        HTTPException: If concepts not found or comparison fails
    """
    try:
        # Get both concepts
        concept1 = postgres_manager.get_concept_by_id(concept_id1)
        concept2 = postgres_manager.get_concept_by_id(concept_id2)
        
        if not concept1 or not concept2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both concepts not found"
            )
        
        # Compare descriptions
        desc1 = postgres_manager.get_descriptions_by_concept_id(concept_id1, active_only=True)
        desc2 = postgres_manager.get_descriptions_by_concept_id(concept_id2, active_only=True)
        
        fsn1 = next((d["term"] for d in desc1 if d["type_id"] == 900000000000003001), "")
        fsn2 = next((d["term"] for d in desc2 if d["type_id"] == 900000000000003001), "")
        
        # Semantic similarity
        semantic_similarity = 0.0
        try:
            if fsn1 and fsn2:
                emb1 = model_manager.generate_embedding(fsn1)
                emb2 = model_manager.generate_embedding(fsn2)
                
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                semantic_similarity = float(similarity)
        except Exception:
            pass
        
        # Graph relationship analysis
        try:
            # Find common ancestors
            parents1 = janusgraph_manager.find_hierarchical_parents(concept_id1, 5)
            parents2 = janusgraph_manager.find_hierarchical_parents(concept_id2, 5)
            
            parents1_ids = {p.get("conceptId") for p in parents1}
            parents2_ids = {p.get("conceptId") for p in parents2}
            common_ancestors = parents1_ids.intersection(parents2_ids)
            
            # Check direct relationship
            relationships1 = postgres_manager.get_relationships_by_concept_id(concept_id1, active_only=True)
            direct_relationship = any(
                r["destination_id"] == concept_id2 or r["source_id"] == concept_id2 
                for r in relationships1
            )
        except Exception:
            common_ancestors = set()
            direct_relationship = False
        
        return {
            "concept_1": {
                "id": concept_id1,
                "fully_specified_name": fsn1,
                "active": concept1.get("active", True)
            },
            "concept_2": {
                "id": concept_id2,
                "fully_specified_name": fsn2,
                "active": concept2.get("active", True)
            },
            "comparison": {
                "semantic_similarity": semantic_similarity,
                "common_ancestors": list(common_ancestors),
                "common_ancestor_count": len(common_ancestors),
                "direct_relationship": direct_relationship,
                "same_module": concept1.get("module_id") == concept2.get("module_id"),
                "both_active": concept1.get("active", True) and concept2.get("active", True)
            },
            "recommendations": {
                "highly_similar": semantic_similarity > 0.8,
                "related": len(common_ancestors) > 0 or direct_relationship,
                "confidence": max(semantic_similarity, 0.5 if common_ancestors else 0.0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Concept comparison failed: {str(e)}"
        )


@router.get("/statistics", summary="Get unified platform statistics")
async def get_platform_statistics(
    postgres_manager = Depends(get_postgres_manager),
    milvus_manager = Depends(get_milvus_manager),
    janusgraph_manager = Depends(get_janusgraph_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive statistics across all databases.
    
    Args:
        postgres_manager: PostgreSQL database manager
        milvus_manager: Milvus database manager
        janusgraph_manager: JanusGraph database manager
        
    Returns:
        Platform-wide statistics and metrics
        
    Raises:
        HTTPException: If statistics retrieval fails
    """
    try:
        stats = {
            "platform": {
                "name": "SNOMED-CT Multi-Modal Data Platform",
                "version": "1.0.0",
                "databases": ["PostgreSQL", "Milvus", "JanusGraph"]
            },
            "databases": {}
        }
        
        # PostgreSQL stats
        try:
            pg_stats = postgres_manager.get_table_counts()
            stats["databases"]["postgresql"] = {
                "status": "connected",
                "tables": pg_stats,
                "total_concepts": pg_stats.get("concepts", 0),
                "total_descriptions": pg_stats.get("descriptions", 0),
                "total_relationships": pg_stats.get("relationships", 0)
            }
        except Exception as e:
            stats["databases"]["postgresql"] = {"status": "error", "error": str(e)}
        
        # Milvus stats
        try:
            mv_stats = milvus_manager.get_collection_stats()
            stats["databases"]["milvus"] = {
                "status": "connected",
                "collection_name": milvus_manager.collection_name,
                "statistics": mv_stats,
                "embedding_dimension": milvus_manager.embedding_dim
            }
        except Exception as e:
            stats["databases"]["milvus"] = {"status": "error", "error": str(e)}
        
        # JanusGraph stats
        try:
            jg_stats = janusgraph_manager.get_graph_statistics()
            stats["databases"]["janusgraph"] = {
                "status": "connected",
                "graph_name": janusgraph_manager.graph_name,
                "statistics": jg_stats
            }
        except Exception as e:
            stats["databases"]["janusgraph"] = {"status": "error", "error": str(e)}
        
        # Summary
        total_concepts = stats["databases"].get("postgresql", {}).get("total_concepts", 0)
        stats["summary"] = {
            "total_concepts": total_concepts,
            "databases_online": sum(1 for db in stats["databases"].values() if db.get("status") == "connected"),
            "databases_total": 3,
            "platform_healthy": all(db.get("status") == "connected" for db in stats["databases"].values())
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get platform statistics: {str(e)}"
        )


# Add logger import at the top
import logging
logger = logging.getLogger(__name__) 