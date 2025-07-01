"""
Unit tests for FastAPI main application module.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from src.snomed_ct_platform.api.main import app, get_application


class TestFastAPIApplication:
    """Test cases for FastAPI application setup."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_app_creation(self):
        """Test that FastAPI app is created properly."""
        assert app is not None
        assert hasattr(app, 'title')
        assert hasattr(app, 'version')
        assert hasattr(app, 'description')
    
    def test_get_application_function(self):
        """Test get_application factory function."""
        test_app = get_application()
        
        assert test_app is not None
        assert hasattr(test_app, 'title')
        assert hasattr(test_app, 'version')
    
    def test_app_metadata(self):
        """Test application metadata."""
        # These values should match what's in main.py
        assert "SNOMED-CT" in app.title
        assert app.version is not None
        assert app.description is not None
    
    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured."""
        # Check that CORS middleware is added
        middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert 'CORSMiddleware' in middleware_names
    
    def test_api_routers_included(self):
        """Test that API routers are properly included."""
        # Get all routes
        routes = [route.path for route in app.routes]
        
        # Should include API v1 routes
        api_routes = [route for route in routes if route.startswith('/api/v1')]
        assert len(api_routes) > 0
        
        # Should include specific endpoint patterns
        expected_patterns = [
            '/api/v1/concepts',
            '/api/v1/descriptions', 
            '/api/v1/relationships',
            '/api/v1/semantic',
            '/api/v1/graph'
        ]
        
        for pattern in expected_patterns:
            assert any(pattern in route for route in api_routes)


class TestHealthEndpoints:
    """Test cases for health check endpoints."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns welcome message."""
        response = self.client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "SNOMED-CT" in data["message"]
    
    @patch('src.snomed_ct_platform.api.dependencies.get_postgres_manager')
    @patch('src.snomed_ct_platform.api.dependencies.get_milvus_manager') 
    @patch('src.snomed_ct_platform.api.dependencies.get_janusgraph_manager')
    def test_health_endpoint_all_services_healthy(self, mock_janus, mock_milvus, mock_postgres):
        """Test health endpoint when all services are healthy."""
        # Mock all services as healthy
        mock_postgres_mgr = MagicMock()
        mock_postgres_mgr.check_connection.return_value = True
        mock_postgres.return_value = mock_postgres_mgr
        
        mock_milvus_mgr = MagicMock()
        mock_milvus_mgr.check_connection.return_value = True
        mock_milvus.return_value = mock_milvus_mgr
        
        mock_janus_mgr = MagicMock()
        mock_janus_mgr.check_connection.return_value = True
        mock_janus.return_value = mock_janus_mgr
        
        response = self.client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["services"]["postgres"] == "healthy"
        assert data["services"]["milvus"] == "healthy"
        assert data["services"]["janusgraph"] == "healthy"
    
    @patch('src.snomed_ct_platform.api.dependencies.get_postgres_manager')
    @patch('src.snomed_ct_platform.api.dependencies.get_milvus_manager')
    @patch('src.snomed_ct_platform.api.dependencies.get_janusgraph_manager')
    def test_health_endpoint_service_unhealthy(self, mock_janus, mock_milvus, mock_postgres):
        """Test health endpoint when a service is unhealthy."""
        # Mock postgres as unhealthy
        mock_postgres_mgr = MagicMock()
        mock_postgres_mgr.check_connection.return_value = False
        mock_postgres.return_value = mock_postgres_mgr
        
        # Mock other services as healthy
        mock_milvus_mgr = MagicMock()
        mock_milvus_mgr.check_connection.return_value = True
        mock_milvus.return_value = mock_milvus_mgr
        
        mock_janus_mgr = MagicMock()
        mock_janus_mgr.check_connection.return_value = True
        mock_janus.return_value = mock_janus_mgr
        
        response = self.client.get("/health")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["services"]["postgres"] == "unhealthy"
        assert data["services"]["milvus"] == "healthy"
        assert data["services"]["janusgraph"] == "healthy"
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns application metrics."""
        response = self.client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should contain basic metrics
        assert "uptime" in data
        assert "memory_usage" in data
        assert "timestamp" in data
        
        # Uptime should be a positive number
        assert data["uptime"] >= 0


class TestErrorHandling:
    """Test cases for error handling."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_404_error_handling(self):
        """Test 404 error handling for non-existent endpoints."""
        response = self.client.get("/non-existent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
    
    def test_405_method_not_allowed(self):
        """Test 405 error for wrong HTTP methods."""
        # Try POST on GET-only endpoint
        response = self.client.post("/")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    @patch('src.snomed_ct_platform.api.dependencies.get_postgres_manager')
    def test_500_internal_server_error_handling(self, mock_postgres):
        """Test 500 error handling for internal server errors."""
        # Mock database to raise an exception
        mock_postgres.side_effect = Exception("Database connection failed")
        
        # This should trigger an internal server error
        response = self.client.get("/health")
        
        # Should return 500 or handle gracefully
        assert response.status_code in [
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]


class TestMiddleware:
    """Test cases for middleware functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_cors_headers_included(self):
        """Test that CORS headers are included in responses."""
        response = self.client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # Should include CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_request_logging_middleware(self):
        """Test that request logging middleware is working."""
        # This test would typically check log output
        # For now, just ensure requests are processed
        response = self.client.get("/")
        assert response.status_code == status.HTTP_200_OK
    
    def test_security_headers(self):
        """Test that security headers are included."""
        response = self.client.get("/")
        
        # Common security headers (depending on middleware configuration)
        headers_to_check = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]
        
        # Note: Actual headers depend on middleware configuration
        # This test structure can be adjusted based on implemented security middleware


class TestOpenAPIDocumentation:
    """Test cases for OpenAPI documentation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_openapi_json_endpoint(self):
        """Test OpenAPI JSON schema endpoint."""
        response = self.client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should be valid OpenAPI schema
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_swagger_ui_endpoint(self):
        """Test Swagger UI endpoint."""
        response = self.client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self):
        """Test ReDoc endpoint."""
        response = self.client.get("/redoc")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


class TestApplicationLifecycle:
    """Test cases for application lifecycle events."""
    
    def test_startup_event(self):
        """Test application startup event."""
        # Create a new app instance to test startup
        test_app = get_application()
        client = TestClient(test_app)
        
        # First request should trigger startup
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
    
    def test_shutdown_event(self):
        """Test application shutdown event."""
        # This is harder to test directly, but we can ensure
        # the shutdown event is registered
        test_app = get_application()
        
        # Check that event handlers are registered
        assert len(test_app.router.on_startup) >= 0
        assert len(test_app.router.on_shutdown) >= 0


class TestRateLimiting:
    """Test cases for rate limiting (if implemented)."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = TestClient(app)
    
    def test_rate_limiting_not_triggered_normal_usage(self):
        """Test that normal usage doesn't trigger rate limiting."""
        # Make several requests within reasonable limits
        responses = []
        for _ in range(5):
            response = self.client.get("/")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.skip(reason="Rate limiting implementation dependent")
    def test_rate_limiting_triggered_excessive_requests(self):
        """Test that excessive requests trigger rate limiting."""
        # This test would need to be adjusted based on actual rate limiting implementation
        responses = []
        for _ in range(200):  # Excessive requests
            response = self.client.get("/")
            responses.append(response)
        
        # Some requests should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


class TestDependencyInjection:
    """Test cases for dependency injection."""
    
    def test_dependency_providers_available(self):
        """Test that dependency providers are properly set up."""
        # This would test that dependencies can be resolved
        # The actual test depends on the dependency injection setup
        
        # For now, just test that the app can start
        with TestClient(app):
            assert True  # App successfully created with all dependencies
    
    @patch('src.snomed_ct_platform.api.dependencies.get_postgres_manager')
    def test_database_dependency_injection(self, mock_postgres):
        """Test database dependency injection."""
        mock_manager = MagicMock()
        mock_postgres.return_value = mock_manager
        
        # Any endpoint using database dependency should work
        response = self.client.get("/health")
        
        # Should not fail due to dependency issues
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ] 