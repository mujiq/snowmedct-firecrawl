#!/usr/bin/env python3
"""
Test Script for SNOMED-CT Graph Pipeline

This script demonstrates the JanusGraph integration functionality by testing:
1. JanusGraph connection and schema setup
2. Concept and relationship ingestion
3. Graph traversal and query capabilities
4. End-to-end graph pipeline execution
"""

import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snomed_ct_platform.graph.graph_pipeline import GraphIngestionPipeline
from snomed_ct_platform.graph.janusgraph_manager import (
    JanusGraphManager, ConceptVertex, RelationshipEdge, RelationshipType
)
from snomed_ct_platform.utils.logging import setup_logging


def test_janusgraph_connection():
    """Test JanusGraph connection and basic operations."""
    print("\n🔗 Testing JanusGraph Connection...")
    
    try:
        # Initialize JanusGraph manager
        janusgraph_manager = JanusGraphManager()
        
        # Test connection
        print("Connecting to JanusGraph...")
        janusgraph_manager.connect()
        
        # Test schema creation
        print("Creating schema...")
        janusgraph_manager.create_schema()
        
        # Test basic operations
        print("Testing basic operations...")
        test_results = janusgraph_manager.test_operations()
        
        if test_results['success']:
            print(f"✅ JanusGraph test passed!")
            print(f"   - Operations tested: {test_results['operations_tested']}")
            print(f"   - Vertices created: {test_results['test_vertices_created']}")
            print(f"   - Edges created: {test_results['test_edges_created']}")
            print(f"   - Graph stats: {test_results['graph_stats']['total_vertices']} vertices, {test_results['graph_stats']['total_edges']} edges")
            return True
        else:
            print(f"❌ JanusGraph test failed: {test_results['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  JanusGraph test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ JanusGraph test failed: {e}")
        return False


def test_graph_modeling():
    """Test graph modeling with SNOMED-CT concepts and relationships."""
    print("\n🧠 Testing Graph Modeling...")
    
    try:
        # Initialize JanusGraph manager
        janusgraph_manager = JanusGraphManager()
        janusgraph_manager.connect()
        janusgraph_manager.create_schema()
        
        # Sample SNOMED-CT concepts
        sample_concepts = [
            ConceptVertex(22298006, "Myocardial infarction (disorder)", True, 900000000000207008, 900000000000074008),
            ConceptVertex(53741008, "Coronary arteriosclerosis (disorder)", True, 900000000000207008, 900000000000074008),
            ConceptVertex(64715009, "Hypertensive disorder (disorder)", True, 900000000000207008, 900000000000074008),
            ConceptVertex(49601007, "Disorder of cardiovascular system (disorder)", True, 900000000000207008, 900000000000074008),
            ConceptVertex(118234003, "Injury to heart (disorder)", True, 900000000000207008, 900000000000074008)
        ]
        
        print(f"Creating {len(sample_concepts)} sample concepts...")
        created_vertices = janusgraph_manager.create_concepts_batch(sample_concepts)
        
        # Sample relationships (IS_A hierarchy)
        sample_relationships = [
            # Myocardial infarction IS_A Injury to heart
            RelationshipEdge(1, 22298006, 118234003, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002),
            # Coronary arteriosclerosis IS_A Disorder of cardiovascular system
            RelationshipEdge(2, 53741008, 49601007, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002),
            # Hypertensive disorder IS_A Disorder of cardiovascular system
            RelationshipEdge(3, 64715009, 49601007, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002),
            # Injury to heart IS_A Disorder of cardiovascular system
            RelationshipEdge(4, 118234003, 49601007, RelationshipType.IS_A.value, 0, True, 900000000000011006, 900000000000451002)
        ]
        
        print(f"Creating {len(sample_relationships)} sample relationships...")
        created_edges = janusgraph_manager.create_relationships_batch(sample_relationships)
        
        # Test queries
        print("Testing graph queries...")
        
        # Test concept lookup
        concept = janusgraph_manager.get_concept_by_id(22298006)
        print(f"   - Concept lookup: {concept['fullySpecifiedName'] if concept else 'Not found'}")
        
        # Test relationship queries
        relationships = janusgraph_manager.get_concept_relationships(22298006, direction='out')
        print(f"   - Outgoing relationships: {len(relationships)}")
        
        # Test hierarchical queries
        parents = janusgraph_manager.find_hierarchical_parents(22298006)
        children = janusgraph_manager.find_hierarchical_children(49601007)
        print(f"   - Parents of MI: {len(parents)}")
        print(f"   - Children of cardiovascular disorder: {len(children)}")
        
        # Clean up test data
        print("Cleaning up test data...")
        for concept in sample_concepts:
            vertices = janusgraph_manager.g.V().hasLabel('Concept').has('conceptId', concept.concept_id).toList()
            for vertex in vertices:
                janusgraph_manager.g.V(vertex).drop().iterate()
        
        print(f"✅ Graph modeling test passed!")
        print(f"   - Vertices created: {len(created_vertices)}")
        print(f"   - Edges created: {len(created_edges)}")
        return True
        
    except ImportError as e:
        print(f"⚠️  Graph modeling test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Graph modeling test failed: {e}")
        return False


def test_graph_pipeline():
    """Test the complete graph ingestion pipeline."""
    print("\n🔄 Testing Graph Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = GraphIngestionPipeline(
            batch_size=50,
            concept_limit=100,
            relationship_limit=200
        )
        
        # Test pipeline operations
        print("Running pipeline test...")
        test_results = pipeline.test_graph_operations()
        
        if test_results['success']:
            print(f"✅ Graph pipeline test passed!")
            
            # Print component status
            components = test_results['connections_status']
            print(f"   - JanusGraph connected: {components['janusgraph_connected']}")
            print(f"   - PostgreSQL connected: {components['postgres_connected']}")
            
            # Print ingestion stats if available
            if 'ingestion_test' in test_results and test_results['ingestion_test']['success']:
                stats = test_results['ingestion_test'].get('pipeline_stats', {})
                print(f"   - Concepts processed: {stats.get('concepts_processed', 0)}")
                print(f"   - Relationships processed: {stats.get('relationships_processed', 0)}")
                print(f"   - Duration: {stats.get('duration_seconds', 0):.2f}s")
            
            return True
        else:
            print(f"❌ Graph pipeline test failed: {test_results['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Graph pipeline test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Graph pipeline test failed: {e}")
        return False


def demonstrate_graph_queries():
    """Demonstrate advanced graph query capabilities."""
    print("\n🔍 Demonstrating Graph Queries...")
    
    try:
        # Initialize pipeline
        pipeline = GraphIngestionPipeline(batch_size=10, concept_limit=50)
        pipeline.setup_connections()
        
        # Sample concept IDs for demonstration
        sample_concept_ids = [22298006, 53741008, 64715009]  # MI, Coronary arteriosclerosis, Hypertension
        
        for concept_id in sample_concept_ids:
            print(f"\n📊 Analyzing concept {concept_id}:")
            
            # Query concept hierarchy
            hierarchy = pipeline.query_concept_hierarchy(concept_id, max_depth=3)
            
            if hierarchy['success']:
                concept = hierarchy['concept']
                print(f"   Concept: {concept.get('fullySpecifiedName', 'Unknown')}")
                print(f"   Parents: {hierarchy['parent_count']}")
                print(f"   Children: {hierarchy['child_count']}")
                
                # Show some parent concepts
                if hierarchy['parents']:
                    print("   Sample parents:")
                    for parent in hierarchy['parents'][:3]:
                        print(f"     - {parent.get('fullySpecifiedName', 'Unknown')}")
                
                # Show some child concepts
                if hierarchy['children']:
                    print("   Sample children:")
                    for child in hierarchy['children'][:3]:
                        print(f"     - {child.get('fullySpecifiedName', 'Unknown')}")
            else:
                print(f"   ❌ Query failed: {hierarchy['error']}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Graph query demo skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Graph query demo failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 SNOMED-CT Graph Pipeline Test Suite")
    print("=" * 50)
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Track test results
    results = []
    
    # Run tests
    print("\n📋 Running Graph Tests...")
    
    # Test 1: JanusGraph connection
    result = test_janusgraph_connection()
    results.append(("JanusGraph Connection", result))
    
    # Test 2: Graph modeling
    result = test_graph_modeling()
    results.append(("Graph Modeling", result))
    
    # Test 3: Complete pipeline
    result = test_graph_pipeline()
    results.append(("Graph Pipeline", result))
    
    # Test 4: Query demonstration
    result = demonstrate_graph_queries()
    results.append(("Graph Queries", result))
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All available graph tests passed!")
        if skipped > 0:
            print("💡 Install missing dependencies to run all tests:")
            print("   pip install gremlinpython psycopg2-binary")
        return 0
    else:
        print(f"\n💥 {failed} graph test(s) failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 