"""
Comprehensive test runner for SNOMED-CT platform unit tests.

This script runs all unit tests with coverage reporting and generates
detailed test reports.
"""

import sys
import os
import subprocess
import pytest
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_tests_with_coverage():
    """Run all tests with coverage reporting."""
    
    print("🚀 Starting comprehensive unit test suite...")
    print("=" * 60)
    
    # Test configuration
    test_args = [
        "-v",  # Verbose output
        "-x",  # Stop on first failure (for debugging)
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--cov=src/snomed_ct_platform",  # Coverage for source code
        "--cov-report=html:tests/htmlcov",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "--cov-report=xml:tests/coverage.xml",  # XML coverage for CI
        "--junit-xml=tests/junit.xml",  # JUnit XML for CI
        "tests/"  # Test directory
    ]
    
    start_time = time.time()
    
    try:
        # Run pytest with coverage
        result = pytest.main(test_args)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Test execution completed in {duration:.2f} seconds")
        
        if result == 0:
            print("✅ All tests passed!")
            print_test_summary()
        else:
            print("❌ Some tests failed!")
            print(f"Exit code: {result}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_individual_test_modules():
    """Run individual test modules to identify issues."""
    
    test_modules = [
        "tests/test_config.py",
        "tests/test_logging.py", 
        "tests/test_rf2_parser.py",
        "tests/test_api_main.py",
        "tests/test_postgres_manager.py"
    ]
    
    print("\n🔍 Running individual test modules...")
    print("-" * 50)
    
    results = {}
    
    for module in test_modules:
        if Path(module).exists():
            print(f"\n📋 Testing {module}...")
            try:
                result = pytest.main(["-v", module])
                results[module] = "✅ PASSED" if result == 0 else "❌ FAILED"
                print(f"   {results[module]}")
            except Exception as e:
                results[module] = f"❌ ERROR: {e}"
                print(f"   {results[module]}")
        else:
            results[module] = "⚠️  FILE NOT FOUND"
            print(f"   {results[module]}")
    
    print("\n📊 Individual Module Results:")
    print("-" * 50)
    for module, result in results.items():
        print(f"{module}: {result}")
    
    return results

def print_test_summary():
    """Print test summary and coverage information."""
    
    print("\n📊 Test Summary")
    print("=" * 60)
    
    # Check if coverage report exists
    coverage_file = Path("tests/htmlcov/index.html")
    if coverage_file.exists():
        print(f"📈 Coverage report generated: {coverage_file}")
        print("   Open in browser to view detailed coverage")
    
    # Check if JUnit XML exists
    junit_file = Path("tests/junit.xml")
    if junit_file.exists():
        print(f"📋 JUnit XML report: {junit_file}")
    
    print("\n🎯 Next Steps:")
    print("1. Review test results and coverage report")
    print("2. Fix any failing tests")
    print("3. Add tests for uncovered code")
    print("4. Aim for >90% test coverage")

def check_test_dependencies():
    """Check if all required test dependencies are available."""
    
    print("🔧 Checking test dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-cov", 
        "pytest-asyncio",
        "coverage"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All test dependencies available")
    return True

def create_test_directories():
    """Create necessary test directories."""
    
    directories = [
        "tests/htmlcov",
        "tests/reports", 
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """Main test execution function."""
    
    print("🧪 SNOMED-CT Platform Test Suite")
    print("=" * 60)
    
    # Create necessary directories
    create_test_directories()
    
    # Check dependencies
    if not check_test_dependencies():
        print("\n❌ Missing test dependencies. Please install them first.")
        return 1
    
    # Run individual modules first for debugging
    print("\n🔍 Phase 1: Individual Module Testing")
    individual_results = run_individual_test_modules()
    
    # Count successful modules
    successful_modules = sum(1 for result in individual_results.values() 
                           if "✅ PASSED" in result)
    total_modules = len(individual_results)
    
    print(f"\n📈 Individual Module Success Rate: {successful_modules}/{total_modules}")
    
    # Run comprehensive test suite
    print("\n🚀 Phase 2: Comprehensive Test Suite")
    comprehensive_result = run_tests_with_coverage()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    if comprehensive_result == 0:
        print("🎉 SUCCESS: All tests passed!")
        print("✨ Your code is well-tested and ready for production!")
    else:
        print("⚠️  Some tests need attention")
        print("🔧 Review the output above and fix failing tests")
    
    return comprehensive_result

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 