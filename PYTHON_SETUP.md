# Python Environment Setup Guide

## 🐍 Installing Python on Windows

### Method 1: Official Python Installer (Recommended)

1. **Download Python**
   - Visit https://python.org/downloads/
   - Download Python 3.8 or later (3.11+ recommended)
   - Choose "Windows installer (64-bit)" for most systems

2. **Install Python**
   - Run the downloaded installer
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
   - Choose "Install Now" or customize installation
   - Verify "pip" is included in the installation

3. **Verify Installation**
   ```powershell
   python --version
   pip --version
   ```

### Method 2: Microsoft Store (Alternative)

1. Open Microsoft Store
2. Search for "Python 3.11" or later
3. Install the official Python from Python Software Foundation
4. Verify installation as above

### Method 3: Chocolatey (For Advanced Users)

```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python
choco install python
```

---

## 🚀 Setting Up the SNOMED-CT API Environment

### 1. Open PowerShell in Project Directory

```powershell
# Navigate to your project
cd C:\Users\gpadmin\Documents\cursor-projects\firecrawl-snowmed-ct
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 4. Test Setup

```powershell
# Run the setup test
python test_api_setup.py
```

Expected output:
```
🚀 Testing SNOMED-CT Multi-Modal Data Platform API Setup
============================================================
Testing package imports...
✅ FastAPI 0.95.2 imported successfully
✅ Uvicorn imported successfully
✅ Pydantic 2.5.0 imported successfully
✅ Pydantic Settings imported successfully

Testing project structure...
✅ Directory exists: src/snomed_ct_platform
✅ File exists: src/snomed_ct_platform/api/main.py
...

🎉 All tests passed! API setup is ready.
```

### 5. Start the API Server

```powershell
# Start the FastAPI development server
python -m uvicorn src.snomed_ct_platform.api.main:app --reload --host localhost --port 8000
```

### 6. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🐳 Prerequisites: Docker Services

Before starting the API, ensure Docker services are running:

```powershell
# Start Docker services
docker-compose up -d

# Check service status
docker-compose ps
```

Required services:
- ✅ **Milvus**: localhost:19530 (gRPC), localhost:9091 (HTTP)
- ✅ **JanusGraph**: localhost:8182 (Gremlin)
- ✅ **Cassandra**: localhost:9042
- ✅ **Elasticsearch**: localhost:9200

---

## 🔧 API Endpoints Available

### Core Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check for all services
- `GET /metrics` - Application metrics

### Concept Endpoints (✅ Implemented)
- `GET /api/v1/concepts/` - List concepts with pagination
- `GET /api/v1/concepts/{concept_id}` - Get specific concept
- `POST /api/v1/concepts/search` - Search concepts by text
- `GET /api/v1/concepts/{concept_id}/descriptions` - Get concept descriptions
- `GET /api/v1/concepts/{concept_id}/relationships` - Get concept relationships

### Placeholder Endpoints (🔄 In Development)
- `/api/v1/descriptions/` - Description queries
- `/api/v1/relationships/` - Relationship queries  
- `/api/v1/semantic/` - Milvus vector search
- `/api/v1/graph/` - JanusGraph traversal
- `/api/v1/search/` - Multi-modal unified search

---

## 🧪 Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Get Concepts
```bash
curl "http://localhost:8000/api/v1/concepts/?page=1&page_size=5"
```

### 3. Search Concepts
```bash
curl -X POST "http://localhost:8000/api/v1/concepts/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "heart", "limit": 10}'
```

---

## 🚨 Troubleshooting

### Python Not Found
```powershell
# Check if Python is in PATH
where python

# If not found, reinstall Python with "Add to PATH" checked
```

### Virtual Environment Issues
```powershell
# If activation fails, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use the batch file:
.\venv\Scripts\activate.bat
```

### Import Errors
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Docker Services Not Running
```powershell
# Start Docker Desktop
# Then start services:
docker-compose up -d

# Check logs if services fail:
docker-compose logs milvus
docker-compose logs janusgraph
```

### Port Already in Use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /F /PID <PID>
```

---

## 📝 Environment Variables

Create a `.env` file in the project root:

```env
# API Settings
HOST=localhost
PORT=8000
DEBUG=true

# Database Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=snomed_ct

MILVUS_HOST=localhost
MILVUS_PORT=19530

JANUSGRAPH_HOST=localhost
JANUSGRAPH_PORT=8182

# API Configuration
DEFAULT_PAGE_SIZE=20
MAX_PAGE_SIZE=1000
DEFAULT_SEARCH_LIMIT=10
MAX_SEARCH_LIMIT=100
```

---

## 🎯 Next Steps

Once the API is running:

1. **Test Database Connections**: Visit `/health` endpoint
2. **Explore API Docs**: Visit `/docs` for interactive documentation
3. **Test Concept Endpoints**: Try searching for medical concepts
4. **Complete Development**: Implement remaining routers for semantic search and graph queries

---

## 📞 Support

If you encounter issues:

1. **Check the test script**: `python test_api_setup.py`
2. **Verify Docker services**: `docker-compose ps`
3. **Check API logs** when running uvicorn
4. **Refer to FastAPI documentation**: https://fastapi.tiangolo.com/

---

**🎉 You're now ready to run the SNOMED-CT Multi-Modal Data Platform API!** 