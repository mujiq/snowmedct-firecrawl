"""
Unit tests for logging utility module.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from loguru import logger

from src.snomed_ct_platform.utils.logging import setup_logging, get_logger


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Remove all existing handlers
        logger.remove()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Remove all handlers and restore default
        logger.remove()
        logger.add(sys.stderr)
    
    def test_setup_logging_default_parameters(self):
        """Test setup_logging with default parameters."""
        with patch.object(logger, 'remove') as mock_remove, \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info') as mock_info:
            
            setup_logging()
            
            # Verify logger.remove() was called
            mock_remove.assert_called_once()
            
            # Verify logger.add() was called for console handler
            assert mock_add.call_count == 1
            
            # Check console handler configuration
            console_call = mock_add.call_args_list[0]
            assert console_call[0][0] == sys.stderr
            assert console_call[1]['level'] == 'INFO'
            assert console_call[1]['colorize'] == True
            
            # Verify info message was logged
            mock_info.assert_called_once_with("Logging initialized with level: INFO")
    
    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info'):
            
            setup_logging(log_level="DEBUG")
            
            # Check that DEBUG level was used
            console_call = mock_add.call_args_list[0]
            assert console_call[1]['level'] == 'DEBUG'
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test setup_logging with file handler."""
        log_file = temp_dir / "test.log"
        
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info'):
            
            setup_logging(log_file=log_file)
            
            # Should have both console and file handlers
            assert mock_add.call_count == 2
            
            # Check console handler (first call)
            console_call = mock_add.call_args_list[0]
            assert console_call[0][0] == sys.stderr
            
            # Check file handler (second call)
            file_call = mock_add.call_args_list[1]
            assert file_call[0][0] == log_file
            assert file_call[1]['level'] == 'INFO'
            assert file_call[1]['rotation'] == '10 MB'
            assert file_call[1]['retention'] == '1 week'
            assert file_call[1]['compression'] == 'zip'
    
    def test_setup_logging_file_directory_creation(self, temp_dir):
        """Test that log file directory is created if it doesn't exist."""
        log_dir = temp_dir / "logs" / "subdir"
        log_file = log_dir / "test.log"
        
        # Ensure directory doesn't exist initially
        assert not log_dir.exists()
        
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add'), \
             patch.object(logger, 'info'):
            
            setup_logging(log_file=log_file)
            
            # Directory should be created
            assert log_dir.exists()
            assert log_dir.is_dir()
    
    def test_setup_logging_custom_rotation_retention(self, temp_dir):
        """Test setup_logging with custom rotation and retention."""
        log_file = temp_dir / "test.log"
        
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info'):
            
            setup_logging(
                log_file=log_file,
                rotation="5 MB",
                retention="2 days"
            )
            
            # Check file handler configuration
            file_call = mock_add.call_args_list[1]
            assert file_call[1]['rotation'] == '5 MB'
            assert file_call[1]['retention'] == '2 days'
    
    def test_setup_logging_format_strings(self):
        """Test that format strings are correctly configured."""
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info'):
            
            setup_logging()
            
            # Check console format
            console_call = mock_add.call_args_list[0]
            console_format = console_call[1]['format']
            
            # Should contain expected format elements
            assert '<green>{time:YYYY-MM-DD HH:mm:ss}</green>' in console_format
            assert '<level>{level: <8}</level>' in console_format
            assert '<cyan>{name}</cyan>' in console_format
            assert '<level>{message}</level>' in console_format
    
    def test_setup_logging_file_format(self, temp_dir):
        """Test file logging format."""
        log_file = temp_dir / "test.log"
        
        with patch.object(logger, 'remove'), \
             patch.object(logger, 'add') as mock_add, \
             patch.object(logger, 'info'):
            
            setup_logging(log_file=log_file)
            
            # Check file format (second call)
            file_call = mock_add.call_args_list[1]
            file_format = file_call[1]['format']
            
            # Should contain expected format elements without color tags
            assert '{time:YYYY-MM-DD HH:mm:ss}' in file_format
            assert '{level: <8}' in file_format
            assert '{name}:{function}:{line}' in file_format
            assert '{message}' in file_format
            
            # Should not contain color tags
            assert '<green>' not in file_format
            assert '<level>' not in file_format


class TestGetLogger:
    """Test cases for get_logger function."""
    
    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a properly bound logger."""
        test_name = "test_module"
        
        with patch.object(logger, 'bind') as mock_bind:
            mock_bound_logger = MagicMock()
            mock_bind.return_value = mock_bound_logger
            
            result = get_logger(test_name)
            
            # Verify logger.bind was called with correct name
            mock_bind.assert_called_once_with(name=test_name)
            
            # Verify bound logger is returned
            assert result == mock_bound_logger
    
    def test_get_logger_different_names(self):
        """Test get_logger with different module names."""
        names = ["module1", "module2", "test.submodule", "__main__"]
        
        with patch.object(logger, 'bind') as mock_bind:
            for name in names:
                get_logger(name)
            
            # Verify bind was called for each name
            expected_calls = [call(name=name) for name in names]
            mock_bind.assert_has_calls(expected_calls)
    
    def test_get_logger_integration(self):
        """Test get_logger integration with actual logger."""
        # This test uses the actual logger to ensure integration works
        module_logger = get_logger("test_module")
        
        # Should be able to log messages
        with patch.object(logger, 'info') as mock_info:
            # The bound logger should have logging methods
            assert hasattr(module_logger, 'info')
            assert hasattr(module_logger, 'debug')
            assert hasattr(module_logger, 'warning')
            assert hasattr(module_logger, 'error')
            assert hasattr(module_logger, 'critical')


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_full_logging_setup_and_usage(self, temp_dir):
        """Test complete logging setup and usage workflow."""
        log_file = temp_dir / "integration_test.log"
        
        # Setup logging
        setup_logging(log_level="DEBUG", log_file=log_file)
        
        # Get a logger
        test_logger = get_logger("integration_test")
        
        # Log some messages (these will actually be logged)
        test_messages = [
            ("debug", "Debug message"),
            ("info", "Info message"),
            ("warning", "Warning message"),
            ("error", "Error message")
        ]
        
        # This test verifies the setup works without errors
        # In a real scenario, you might want to capture and verify log output
        try:
            for level, message in test_messages:
                getattr(test_logger, level)(message)
            
            # If we get here without exceptions, the integration works
            assert True
        except Exception as e:
            pytest.fail(f"Logging integration failed: {e}")
    
    def test_logger_hierarchy(self):
        """Test that loggers maintain proper hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        grandchild_logger = get_logger("parent.child.grandchild")
        
        # All should be logger instances
        assert parent_logger is not None
        assert child_logger is not None
        assert grandchild_logger is not None
        
        # Should be able to log from all levels
        try:
            parent_logger.info("Parent message")
            child_logger.info("Child message")
            grandchild_logger.info("Grandchild message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger hierarchy test failed: {e}")
    
    def test_logging_performance(self):
        """Test logging performance with multiple loggers."""
        import time
        
        loggers = [get_logger(f"perf_test_{i}") for i in range(100)]
        
        start_time = time.time()
        
        for i, logger_instance in enumerate(loggers):
            logger_instance.info(f"Performance test message {i}")
        
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        duration = end_time - start_time
        assert duration < 5.0, f"Logging performance test took too long: {duration}s" 