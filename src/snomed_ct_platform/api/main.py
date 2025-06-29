"""
SNOMED-CT Multi-Modal Data Platform API

FastAPI application providing unified access to PostgreSQL, Milvus, and JanusGraph
for comprehensive SNOMED-CT terminology queries.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from .routers import concepts, descriptions, relationships, semantic_search, graph_queries, unified_search
from .dependencies import get_database_managers
from .middleware import add_process_time_header, LoggingMiddleware
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting SNOMED-CT Multi-Modal Data Platform API")
    
    # Startup
    try:
        # Initialize database connections
        managers = await get_database_managers()
        logger.info("Database connections initialized successfully")
        
        # Store managers in app state
        app.state.managers = managers
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down SNOMED-CT Multi-Modal Data Platform API")
        if hasattr(app.state, 'managers'):
            # Close database connections
            if 'postgres' in app.state.managers:
                app.state.managers['postgres'].close()
            if 'milvus' in app.state.managers:
                app.state.managers['milvus'].close()
            if 'janusgraph' in app.state.managers:
                app.state.managers['janusgraph'].close()
        logger.info("Database connections closed")

# Create FastAPI application
app = FastAPI(
    title="SNOMED-CT Multi-Modal Data Platform",
    description="A comprehensive API for querying SNOMED-CT terminology data across PostgreSQL, Milvus, and JanusGraph databases",
    version="1.0.0",
    contact={
        "name": "SNOMED-CT Platform Team",
        "url": "https://github.com/your-org/snomed-ct-platform",
        "email": "contact@your-org.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.middleware("http")(add_process_time_header)

# Include routers
app.include_router(
    concepts.router,
    prefix="/api/v1/concepts",
    tags=["concepts"],
    dependencies=[Depends(get_database_managers)]
)

app.include_router(
    descriptions.router,
    prefix="/api/v1/descriptions",
    tags=["descriptions"],
    dependencies=[Depends(get_database_managers)]
)

app.include_router(
    relationships.router,
    prefix="/api/v1/relationships",
    tags=["relationships"],
    dependencies=[Depends(get_database_managers)]
)

app.include_router(
    semantic_search.router,
    prefix="/api/v1/semantic",
    tags=["semantic-search"],
    dependencies=[Depends(get_database_managers)]
)

app.include_router(
    graph_queries.router,
    prefix="/api/v1/graph",
    tags=["graph-queries"],
    dependencies=[Depends(get_database_managers)]
)

app.include_router(
    unified_search.router,
    prefix="/api/v1/search",
    tags=["unified-search"],
    dependencies=[Depends(get_database_managers)]
)

@app.get("/", summary="Root endpoint")
async def root() -> Dict[str, str]:
    """Welcome endpoint with basic API information."""
    return {
        "message": "Welcome to SNOMED-CT Multi-Modal Data Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }

@app.get("/health", summary="Health check endpoint")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Check database connections
        managers = app.state.managers
        health_status = {
            "status": "healthy",
            "databases": {
                "postgres": "unknown",
                "milvus": "unknown",
                "janusgraph": "unknown"
            }
        }
        
        # Check PostgreSQL
        try:
            if 'postgres' in managers and managers['postgres'].test_connection():
                health_status["databases"]["postgres"] = "healthy"
            else:
                health_status["databases"]["postgres"] = "unhealthy"
        except Exception:
            health_status["databases"]["postgres"] = "error"
            
        # Check Milvus
        try:
            if 'milvus' in managers and managers['milvus'].test_connection():
                health_status["databases"]["milvus"] = "healthy"
            else:
                health_status["databases"]["milvus"] = "unhealthy"
        except Exception:
            health_status["databases"]["milvus"] = "error"
            
        # Check JanusGraph
        try:
            if 'janusgraph' in managers and managers['janusgraph'].test_connection():
                health_status["databases"]["janusgraph"] = "healthy"
            else:
                health_status["databases"]["janusgraph"] = "unhealthy"
        except Exception:
            health_status["databases"]["janusgraph"] = "error"
            
        # Overall status
        unhealthy_dbs = [db for db, status in health_status["databases"].items() 
                        if status in ["unhealthy", "error"]]
        if unhealthy_dbs:
            health_status["status"] = "degraded"
            health_status["issues"] = unhealthy_dbs
            
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )

@app.get("/metrics", summary="Application metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get application metrics for monitoring."""
    try:
        managers = app.state.managers
        metrics = {
            "timestamp": "2024-01-01T00:00:00Z",  # Will be replaced with actual timestamp
            "requests": {
                "total": 0,  # Will be tracked by middleware
                "errors": 0,
                "avg_response_time": 0.0
            },
            "databases": {}
        }
        
        # Database-specific metrics
        if 'postgres' in managers:
            metrics["databases"]["postgres"] = {
                "active_connections": 1,  # Will be real connection count
                "total_concepts": 0,      # Will query actual count
                "total_descriptions": 0,
                "total_relationships": 0
            }
            
        if 'milvus' in managers:
            metrics["databases"]["milvus"] = {
                "collections": 1,         # Will query actual collections
                "total_embeddings": 0,    # Will query actual count
                "index_status": "ready"
            }
            
        if 'janusgraph' in managers:
            metrics["databases"]["janusgraph"] = {
                "vertices": 0,            # Will query actual count
                "edges": 0,
                "graph_size": "0MB"
            }
            
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collection failed"
        )

def custom_openapi():
    """Custom OpenAPI schema generation."""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://your-org.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 