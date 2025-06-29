#!/usr/bin/env python3
"""
Test Script for SNOMED-CT Embedding Pipeline

This script demonstrates the embedding pipeline functionality by testing:
1. Embedding model loading and inference
2. Milvus collection setup and operations
3. End-to-end embedding generation and storage
"""

import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snomed_ct_platform.embeddings.embedding_pipeline import EmbeddingPipeline
from snomed_ct_platform.embeddings.model_manager import EmbeddingModelManager, get_recommended_model
from snomed_ct_platform.database.milvus_manager import MilvusManager
from snomed_ct_platform.utils.logging import setup_logging


def test_model_loading():
    """Test embedding model loading and inference."""
    print("\n🧠 Testing Embedding Model Loading...")
    
    try:
        # Get recommended model for clinical use
        model_name = get_recommended_model('clinical')
        print(f"Using model: {model_name}")
        
        # Initialize model manager
        model_manager = EmbeddingModelManager(
            model_name=model_name,
            device='cpu'  # Use CPU for testing
        )
        
        # Test model loading
        print("Loading model...")
        model_manager.load_model()
        
        # Test embedding generation
        print("Testing embedding generation...")
        test_results = model_manager.test_embedding_generation()
        
        if test_results['success']:
            print(f"✅ Model test passed!")
            print(f"   - Embedding dimension: {test_results['embedding_dimension']}")
            print(f"   - Samples processed: {test_results['num_samples']}")
            print(f"   - Sample similarity: {test_results.get('sample_similarity', 'N/A'):.4f}")
            return True
        else:
            print(f"❌ Model test failed: {test_results['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Model test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_milvus_operations():
    """Test Milvus database operations."""
    print("\n🗄️  Testing Milvus Operations...")
    
    try:
        # Initialize Milvus manager
        milvus_manager = MilvusManager(
            embedding_dim=768  # Standard BERT dimension
        )
        
        # Test connection
        print("Connecting to Milvus...")
        milvus_manager.connect()
        
        # Test collection operations
        print("Testing collection operations...")
        milvus_manager.create_collection(drop_existing=True)
        milvus_manager.create_index()
        milvus_manager.load_collection()
        
        # Test CRUD operations
        print("Testing CRUD operations...")
        test_results = milvus_manager.test_operations()
        
        if test_results['success']:
            print(f"✅ Milvus test passed!")
            print(f"   - Operations tested: {test_results['operations_tested']}")
            print(f"   - Test data size: {test_results['test_data_size']}")
            print(f"   - Search results: {test_results['search_results_count']}")
            return True
        else:
            print(f"❌ Milvus test failed: {test_results['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Milvus test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Milvus test failed: {e}")
        return False


def test_embedding_pipeline():
    """Test the complete embedding pipeline."""
    print("\n🔄 Testing Complete Embedding Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = EmbeddingPipeline(
            device='cpu',
            embedding_batch_size=16,
            storage_batch_size=100
        )
        
        # Test pipeline
        print("Running pipeline test...")
        test_results = pipeline.test_pipeline()
        
        if test_results['success']:
            print(f"✅ Pipeline test passed!")
            
            # Print component status
            components = test_results['components_status']
            print(f"   - Model loaded: {components['model_loaded']}")
            print(f"   - Milvus connected: {components['milvus_connected']}")
            print(f"   - PostgreSQL connected: {components['postgres_connected']}")
            
            # Print pipeline stats if available
            if 'pipeline_test' in test_results and test_results['pipeline_test']['success']:
                stats = test_results['pipeline_test'].get('pipeline_stats', {})
                print(f"   - Concepts processed: {stats.get('concepts_processed', 0)}")
                print(f"   - Embeddings generated: {stats.get('embeddings_generated', 0)}")
                print(f"   - Embeddings stored: {stats.get('embeddings_stored', 0)}")
                print(f"   - Duration: {stats.get('duration_seconds', 0):.2f}s")
            
            return True
        else:
            print(f"❌ Pipeline test failed: {test_results['error']}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Pipeline test skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def run_sample_embedding_demo():
    """Run a sample embedding generation demo with predefined concepts."""
    print("\n🎯 Running Sample Embedding Demo...")
    
    try:
        # Sample SNOMED-CT concepts for demo
        sample_concepts = [
            {"id": 22298006, "fully_specified_name": "Myocardial infarction", "active": True, "module_id": 900000000000207008},
            {"id": 44054006, "fully_specified_name": "Type 2 diabetes mellitus", "active": True, "module_id": 900000000000207008},
            {"id": 38341003, "fully_specified_name": "Hypertensive disorder", "active": True, "module_id": 900000000000207008},
            {"id": 233604007, "fully_specified_name": "Pneumonia", "active": True, "module_id": 900000000000207008},
            {"id": 125605004, "fully_specified_name": "Fracture of bone", "active": True, "module_id": 900000000000207008}
        ]
        
        print(f"Sample concepts: {len(sample_concepts)}")
        for concept in sample_concepts:
            print(f"  - {concept['id']}: {concept['fully_specified_name']}")
        
        # Initialize components separately for demo
        print("\nInitializing components...")
        
        # Model manager
        model_manager = EmbeddingModelManager(device='cpu')
        model_manager.load_model()
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        terms = [c['fully_specified_name'] for c in sample_concepts]
        embeddings = model_manager.generate_embeddings_batch(terms, show_progress=True)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   - Embedding dimension: {len(embeddings[0])}")
        print(f"   - Sample embedding stats:")
        print(f"     • Mean: {embeddings[0].mean():.4f}")
        print(f"     • Std: {embeddings[0].std():.4f}")
        print(f"     • Min: {embeddings[0].min():.4f}")
        print(f"     • Max: {embeddings[0].max():.4f}")
        
        # Test similarity
        if len(embeddings) >= 2:
            import numpy as np
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"   - Similarity between first two concepts: {similarity:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Demo skipped - dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 SNOMED-CT Embedding Pipeline Test Suite")
    print("=" * 50)
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Track test results
    results = []
    
    # Run tests
    print("\n📋 Running Tests...")
    
    # Test 1: Model loading
    result = test_model_loading()
    results.append(("Model Loading", result))
    
    # Test 2: Milvus operations
    result = test_milvus_operations()
    results.append(("Milvus Operations", result))
    
    # Test 3: Complete pipeline
    result = test_embedding_pipeline()
    results.append(("Complete Pipeline", result))
    
    # Test 4: Sample demo
    result = run_sample_embedding_demo()
    results.append(("Sample Demo", result))
    
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
        print("\n🎉 All available tests passed!")
        if skipped > 0:
            print("💡 Install missing dependencies to run all tests:")
            print("   pip install torch transformers sentence-transformers pymilvus")
        return 0
    else:
        print(f"\n💥 {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 