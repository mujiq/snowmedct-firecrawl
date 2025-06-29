"""
API routers package for SNOMED-CT Multi-Modal Data Platform.
"""

# Import all routers for easy access
from . import concepts
from . import descriptions  
from . import relationships
from . import semantic_search
from . import graph_queries
from . import unified_search

__all__ = [
    "concepts",
    "descriptions", 
    "relationships",
    "semantic_search",
    "graph_queries",
    "unified_search"
] 