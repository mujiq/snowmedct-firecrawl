# SNOMED-CT Multi-Modal Data Platform

A comprehensive data platform for ingesting, processing, and querying SNOMED-CT terminology data across multiple database systems: PostgreSQL for structured data, Milvus for vector embeddings, and JanusGraph for graph relationships.

## 🚀 Current Status

**Phase 1-5 Complete**: The project now has full support for PostgreSQL, Milvus, and JanusGraph integration with comprehensive data ingestion pipelines.

### ✅ Completed Features

- **RF2 Parser**: Complete parsing of SNOMED-CT Release Format 2 files
- **PostgreSQL Integration**: Structured storage with optimized schema and indexes
- **Milvus Vector Database**: Semantic search with biomedical embeddings
- **JanusGraph Property Graph**: Relationship modeling and graph traversal
- **Embedding Pipeline**: BioBERT, ClinicalBERT, and PubMedBERT support
- **Multi-Modal Pipelines**: Coordinated ingestion across all three databases
- **Batch Processing**: Configurable batch sizes with error handling
- **Comprehensive Logging**: Real-time progress tracking and statistics

## 🏗️ Architecture

The platform uses a multi-modal approach to store and query SNOMED-CT data:

1. **PostgreSQL**: Stores structured concept, description, and relationship data
2. **Milvus**: Stores vector embeddings for semantic similarity search
3. **JanusGraph**: Models concept relationships as a property graph

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Milvus 2.3+
- JanusGraph 0.6+
- UMLS License (for SNOMED-CT data access)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd firecrawl-snowmed-ct
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp env.template .env
# Edit .env with your database configurations
```

## 🗄️ Database Setup

### PostgreSQL
```bash
# Create database
createdb snomed_ct

# The schema will be created automatically by the pipeline
```

### Milvus
```bash
# Install Milvus (Docker)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# Collections will be created automatically by the pipeline
```

### JanusGraph
```bash
# Install JanusGraph
wget https://github.com/JanusGraph/janusgraph/releases/download/v0.6.3/janusgraph-0.6.3.zip
unzip janusgraph-0.6.3.zip
cd janusgraph-0.6.3
bin/janusgraph-server.sh start

# Schema will be created automatically by the pipeline
```

## 📥 Data Ingestion

### Step 1: Obtain SNOMED-CT Data

1. Register for a UMLS license at: https://uts.nlm.nih.gov/license.html
2. Download SNOMED-CT International Edition RF2 files
3. Extract to `data/rf2/` directory

### Step 2: Run Data Ingestion Pipeline

```bash
# Full pipeline (PostgreSQL + Milvus + JanusGraph)
python src/snomed_ct_platform/pipeline.py --data-dir data/rf2/

# PostgreSQL only
python src/snomed_ct_platform/pipeline.py --data-dir data/rf2/ --postgres-only

# Embeddings pipeline (requires PostgreSQL data)
python src/snomed_ct_platform/embeddings/embedding_pipeline.py

# Graph pipeline (requires PostgreSQL data)
python src/snomed_ct_platform/graph/graph_pipeline.py
```

### Step 3: Verify Data Ingestion

```bash
# Test all pipelines
python test_embedding_pipeline.py
python test_graph_pipeline.py

# Check data integrity
python src/snomed_ct_platform/pipeline.py --validate-only
```

## 🔍 Usage Examples

### PostgreSQL Queries

```python
from snomed_ct_platform.database.postgres_manager import PostgresManager

# Initialize manager
postgres_manager = PostgresManager()
postgres_manager.connect()

# Get active concepts
concepts = postgres_manager.get_active_concepts(limit=100)

# Search descriptions
descriptions = postgres_manager.search_descriptions("myocardial infarction")
```

### Semantic Search with Milvus

```python
from snomed_ct_platform.embeddings.embedding_pipeline import EmbeddingPipeline

# Initialize pipeline
pipeline = EmbeddingPipeline()
pipeline.setup_components()

# Search similar concepts
similar = pipeline.milvus_manager.search_similar(
    query_embedding=embedding_vector,
    limit=10
)
```

### Graph Traversal with JanusGraph

```python
from snomed_ct_platform.graph.graph_pipeline import GraphIngestionPipeline

# Initialize pipeline
pipeline = GraphIngestionPipeline()
pipeline.setup_connections()

# Query concept hierarchy
hierarchy = pipeline.query_concept_hierarchy(22298006)  # Myocardial infarction
print(f"Parents: {hierarchy['parent_count']}")
print(f"Children: {hierarchy['child_count']}")
```

## 📊 Performance Metrics

### Current Benchmarks
- **PostgreSQL**: 1,000 records/batch, ~50,000 concepts/minute
- **Milvus**: 32 embeddings/batch, ~1,000 embeddings/minute
- **JanusGraph**: 100 vertices/batch, ~500 vertices/minute
- **Memory Usage**: ~2GB peak for full ingestion
- **Storage**: ~10GB for complete SNOMED-CT dataset

## 🧪 Testing

### Run All Tests
```bash
# Test PostgreSQL and Milvus pipelines
python test_embedding_pipeline.py

# Test JanusGraph pipeline
python test_graph_pipeline.py

# Test individual components
python -m pytest tests/
```

### Test Modes
- **Fast Test**: Limited dataset (100 concepts)
- **Integration Test**: Cross-database validation
- **Performance Test**: Benchmarking with timing

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `POSTGRES_USER` | PostgreSQL user | postgres |
| `POSTGRES_PASSWORD` | PostgreSQL password | password |
| `POSTGRES_DB` | PostgreSQL database | snomed_ct |
| `MILVUS_HOST` | Milvus host | localhost |
| `MILVUS_PORT` | Milvus port | 19530 |
| `JANUSGRAPH_HOST` | JanusGraph host | localhost |
| `JANUSGRAPH_PORT` | JanusGraph port | 8182 |
| `EMBEDDING_MODEL` | Embedding model | clinicalbert |
| `BATCH_SIZE` | Processing batch size | 1000 |

### Embedding Models

| Model | Description | Dimension |
|-------|-------------|-----------|
| `biobert` | General biomedical BERT | 768 |
| `clinicalbert` | Clinical text BERT | 768 |
| `pubmedbert` | PubMed abstract BERT | 768 |
| `biobert-sentence` | Sentence-optimized BioBERT | 768 |

## 📈 Monitoring

### Logging
- Real-time progress tracking
- Comprehensive error reporting
- Performance metrics collection
- Batch-level recovery information

### Metrics
- Records processed per second
- Database query performance
- Memory usage patterns
- Error rates and types

## 🚧 Roadmap

### Phase 6-7: API Development
- [ ] FastAPI application framework
- [ ] Multi-modal search endpoints
- [ ] GraphQL integration
- [ ] Authentication and authorization

### Phase 8-9: Production Deployment
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Monitoring and alerting
- [ ] Performance optimization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SNOMED International for SNOMED-CT terminology
- National Library of Medicine for UMLS access
- Hugging Face for transformer models
- Milvus community for vector database support
- JanusGraph community for graph database capabilities

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` directory
- Review test examples in `test_*.py` files

---

**Note**: This project requires a UMLS license for SNOMED-CT data access. Please ensure compliance with SNOMED International licensing terms. 