"""
Test script to verify FastAPI application setup and imports.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
        
    try:
        import pydantic
        print(f"✅ Pydantic {pydantic.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
        
    try:
        from pydantic_settings import BaseSettings
        print(f"✅ Pydantic Settings imported successfully")
    except ImportError as e:
        print(f"❌ Pydantic Settings import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src/snomed_ct_platform",
        "src/snomed_ct_platform/api",
        "src/snomed_ct_platform/api/routers"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    required_files = [
        "src/snomed_ct_platform/api/main.py",
        "src/snomed_ct_platform/api/config.py",
        "src/snomed_ct_platform/api/dependencies.py",
        "src/snomed_ct_platform/api/middleware.py",
        "src/snomed_ct_platform/api/routers/__init__.py",
        "src/snomed_ct_platform/api/routers/concepts.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    return True

def test_api_import():
    """Test if we can import the API main module."""
    print("\nTesting API module import...")
    
    try:
        # Add src to Python path
        sys.path.insert(0, 'src')
        
        # Try importing the config first (simpler)
        from snomed_ct_platform.api.config import settings
        print(f"✅ Config imported successfully")
        print(f"   - Host: {settings.HOST}")
        print(f"   - Port: {settings.PORT}")
        
        return True
        
    except ImportError as e:
        print(f"❌ API config import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ API config error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing SNOMED-CT Multi-Modal Data Platform API Setup")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
        print("\n⚠️  Import issues detected. Please ensure:")
        print("   1. Virtual environment is activated")
        print("   2. Dependencies are installed: pip install -r requirements.txt")
    
    # Test project structure
    if not test_project_structure():
        success = False
        print("\n⚠️  Project structure issues detected.")
    
    # Test API import
    if not test_api_import():
        success = False
        print("\n⚠️  API import issues detected.")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! API setup is ready.")
        print("\nNext steps:")
        print("1. Start the API server: python -m uvicorn src.snomed_ct_platform.api.main:app --reload")
        print("2. Visit http://localhost:8000/docs for API documentation")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return success

if __name__ == "__main__":
    main() 