## Project Specifications: Ingestion of SNOMED-CT into a Multi-Modal Data Platform

**Project Cursors:** `snomed-ct-ingestion`, `janusgraph`, `milvus`, `postgres`, `data-pipeline`

### 1\. Executive Summary

This project outlines the comprehensive methodology for downloading the complete SNOMED-CT terminology, processing it, and storing it across a multi-modal data platform comprising PostgreSQL for structured data, Milvus for vector embeddings, and JanusGraph for graph-based entities. This will create a powerful and queryable representation of SNOMED-CT, enabling advanced analytics, semantic search, and graph-based exploration of medical concepts.

The core of this project involves parsing the SNOMED-CT Release Format 2 (RF2) files, structuring the terminological data for relational storage, generating high-quality embeddings for semantic representation, and modeling the intricate relationships between medical concepts as a property graph.

**Note on Firecrawl:** The initial proposal to use Firecrawl to "crawl" SNOMED-CT has been revised. SNOMED-CT is available as a downloadable set of structured text files (RF2 format), making a web crawler unsuitable for initial data acquisition. Instead, this project will utilize a dedicated Python-based parsing solution to process these files directly.

### 2\. Core Technologies

  * **SNOMED-CT:** A comprehensive, multilingual clinical healthcare terminology that provides a standardized vocabulary for clinical documentation and reporting.
  * **PostgreSQL:** A powerful, open-source object-relational database system used to store the structured, tabular data of SNOMED-CT concepts, descriptions, and relationships.
  * **Milvus:** A highly scalable, open-source vector database designed for efficient similarity searches on massive-scale embedding vectors. It will store the generated embeddings of SNOMED-CT concepts.
  * **JanusGraph:** A scalable, distributed graph database optimized for storing and querying large-scale graphs. It will be used to represent and explore the complex hierarchical and poly-hierarchical relationships within SNOMED-CT.
  * **Python:** The primary programming language for the data processing pipeline, including parsing, data transformation, embedding generation, and database ingestion.

### 3\. Project Phases & Step-by-Step Implementation

#### **Phase 1: SNOMED-CT Data Acquisition and Initial Processing**

1.  **Obtain SNOMED-CT Release Files:**

      * **Action:** Register for a UMLS (Unified Medical Language System) license from the National Library of Medicine (NLM).
      * **Source:** Download the latest release of the SNOMED-CT International Edition from the NLM UMLS Terminology Services (UTS).
      * **Format:** The download will be a zip file containing the SNOMED-CT data in Release Format 2 (RF2), which consists of tab-delimited text files.

2.  **Develop a Python-based RF2 Parser:**

      * **Action:** Create a Python script to unzip and parse the RF2 files. The key files to process are:
          * `sct2_Concept_Snapshot_INT...txt`: Contains core concept information (Concept ID, active status, etc.).
          * `sct2_Description_Snapshot-en_INT...txt`: Contains various descriptions for each concept (Fully Specified Name, Synonym, Preferred Term).
          * `sct2_Relationship_Snapshot_INT...txt`: Defines the relationships between concepts (e.g., 'Is a', 'Finding site').
      * **Libraries:** `pandas` for efficient parsing of the large text files.

#### **Phase 2: Structured Data Storage in PostgreSQL**

1.  **Define PostgreSQL Schema:**

      * **Action:** Design a relational schema to store the parsed SNOMED-CT data. The following tables are recommended as a starting point:

        **`concepts` Table:**

        ```sql
        CREATE TABLE concepts (
            id BIGINT PRIMARY KEY,
            effective_time TIMESTAMP,
            active BOOLEAN,
            module_id BIGINT,
            definition_status_id BIGINT
        );
        ```

        **`descriptions` Table:**

        ```sql
        CREATE TABLE descriptions (
            id BIGINT PRIMARY KEY,
            effective_time TIMESTAMP,
            active BOOLEAN,
            module_id BIGINT,
            concept_id BIGINT REFERENCES concepts(id),
            language_code VARCHAR(2),
            type_id BIGINT,
            term TEXT,
            case_significance_id BIGINT
        );
        ```

        **`relationships` Table:**

        ```sql
        CREATE TABLE relationships (
            id BIGINT PRIMARY KEY,
            effective_time TIMESTAMP,
            active BOOLEAN,
            module_id BIGINT,
            source_id BIGINT REFERENCES concepts(id),
            destination_id BIGINT REFERENCES concepts(id),
            relationship_group INTEGER,
            type_id BIGINT,
            characteristic_type_id BIGINT,
            modifier_id BIGINT
        );
        ```

2.  **Data Ingestion into PostgreSQL:**

      * **Action:** Develop a Python script to connect to the PostgreSQL database and populate the defined tables with the parsed data from the RF2 files.
      * **Libraries:** `psycopg2` or `SQLAlchemy` for database connection and data insertion.

#### **Phase 3: Embedding Generation and Storage in Milvus**

1.  **Select a Pre-trained Embedding Model:**

      * **Recommendation:** Utilize a pre-trained model specialized for biomedical text to generate meaningful embeddings.
      * **Primary Candidate:** **`BioBERT`** or **`ClinicalBERT`**. These models have been trained on large biomedical corpora and are well-suited for understanding the nuances of medical terminology.
      * **Alternative:** Explore models from the Hugging Face Model Hub that are specifically fine-tuned on SNOMED-CT or related tasks.

2.  **Generate Embeddings:**

      * **Action:** For each active concept in the `concepts` table, generate a vector embedding.
      * **Input Text for Embedding:** Use the 'Fully Specified Name' (FSN) from the `descriptions` table as the primary input text for the embedding model. Consider concatenating the FSN with the 'Preferred Term' for a more comprehensive representation.
      * **Libraries:** `transformers` (from Hugging Face) to load the pre-trained model and tokenizer, and `torch` or `tensorflow` for the embedding generation process.

3.  **Define Milvus Collection Schema:**

      * **Action:** Create a collection in Milvus to store the SNOMED-CT embeddings. The schema should include:
          * A primary key field to store the `concept_id`.
          * The vector field to store the generated embedding.
          * Optionally, a field for the FSN for easier identification.

    <!-- end list -->

    ```python
    from pymilvus import CollectionSchema, FieldSchema, DataType

    fields = [
        FieldSchema(name="concept_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="fully_specified_name", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768) # Dimension depends on the chosen model
    ]
    schema = CollectionSchema(fields, description="SNOMED-CT Concept Embeddings")
    ```

4.  **Ingest Embeddings into Milvus:**

      * **Action:** As embeddings are generated, insert them into the Milvus collection along with their corresponding `concept_id` and FSN.
      * **Libraries:** `pymilvus` for interacting with the Milvus database.
      * **Indexing:** Create an appropriate index (e.g., `HNSW` or `IVF_FLAT`) on the `embedding` field to enable efficient similarity searches.

#### **Phase 4: Graph Entity Storage in JanusGraph**

1.  **Define JanusGraph Schema:**

      * **Action:** Define the schema for vertices and edges in JanusGraph to represent the SNOMED-CT graph.

        **Vertex Labels:**

          * `Concept`: Represents a SNOMED-CT concept.

        **Vertex Properties:**

          * `conceptId` (BIGINT, indexed)
          * `fullySpecifiedName` (String)
          * `active` (Boolean)
          * `definitionStatus` (String)

        **Edge Labels:**

          * `IS_A`: For hierarchical relationships (e.g., `116680003 |Is a|`).
          * Other relationship types found in the `relationships` table (e.g., `FINDING_SITE`, `CAUSATIVE_AGENT`).

        **Edge Properties:**

          * `relationshipGroup` (Integer)
          * `active` (Boolean)

2.  **Data Ingestion into JanusGraph:**

      * **Action:** Develop a Python script to connect to the JanusGraph server and build the graph.
      * **Process:**
        1.  Iterate through the `concepts` table in PostgreSQL and create a `Concept` vertex for each active concept in JanusGraph.
        2.  Iterate through the `relationships` table in PostgreSQL. For each active relationship, create a directed edge between the `source_id` and `destination_id` vertices, using the relationship type as the edge label.
      * **Libraries:** `gremlinpython` to execute Gremlin queries against the JanusGraph server.

### 4\. Infrastructure and Deployment

  * **Database Setup:** Provision and configure instances of PostgreSQL, Milvus, and JanusGraph. These can be deployed on-premises, in the cloud (e.g., AWS, GCP, Azure), or using Docker containers for local development and testing.
  * **Data Pipeline Orchestration:** For production environments, consider using a workflow management tool like Apache Airflow or Prefect to orchestrate the entire data ingestion pipeline, from downloading the SNOMED-CT files to populating all three databases.

### 5\. API and Query Layer

Once the data is ingested, a unified API layer (e.g., using FastAPI or Flask in Python) should be developed to provide a single point of access to the multi-modal data store. This API should expose endpoints for:

  * **Structured Queries:** Fetching concept details and relationships from PostgreSQL.
  * **Semantic Search:** Performing vector similarity searches on SNOMED-CT concepts in Milvus (e.g., "find concepts similar to 'myocardial infarction'").
  * **Graph Traversal:** Executing Gremlin queries on JanusGraph to explore concept hierarchies, find common ancestors, and analyze complex relationships.

This comprehensive project will result in a robust and versatile SNOMED-CT data platform, unlocking new possibilities for clinical data analysis, application development, and research.