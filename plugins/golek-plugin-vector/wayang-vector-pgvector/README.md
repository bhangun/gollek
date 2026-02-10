# Wayang PostgreSQL Vector Store

The Wayang PostgreSQL Vector Store provides a production-ready vector storage implementation using PostgreSQL with the pgvector extension.

## Overview

This module implements the Wayang VectorStore interface using PostgreSQL as the backend, leveraging the pgvector extension for efficient vector similarity search operations.

## Features

- **Production-ready**: Built for high-performance, scalable vector storage
- **PostgreSQL Backend**: Leverages the reliability and features of PostgreSQL
- **pgvector Extension**: Uses the official pgvector extension for vector operations
- **Reactive**: Built with Quarkus and Mutiny for reactive, non-blocking operations
- **Metadata Filtering**: Supports filtering by metadata attributes
- **ACID Transactions**: Ensures data consistency with PostgreSQL transactions
- **Configurable Indexing**: Supports HNSW and IVFFLAT indexing strategies

## Architecture

The implementation uses:
- **Hibernate Reactive**: For reactive database operations
- **pgvector**: For vector similarity operations
- **JSONB**: For flexible metadata storage
- **GIN Indexes**: For fast metadata queries

## Configuration

The following configuration properties are available:

```properties
# Vector dimension (default: 1536)
wayang.vector.pgvector.dimension=1536

# Index type: hnsw or ivfflat (default: hnsw)
wayang.vector.pgvector.index.type=hnsw

# PostgreSQL connection settings
quarkus.datasource.reactive.url=postgresql://localhost:5432/wayang
quarkus.datasource.username=user
quarkus.datasource.password=password
```

## Usage

### Injection

```java
@Inject
PgVectorStore vectorStore;
```

### Storing Vectors

```java
VectorEntry entry = new VectorEntry(
    "unique-id",
    List.of(0.1f, 0.2f, 0.3f), // Vector embedding
    "Content text",
    Map.of("category", "example", "source", "api")
);

Uni<Void> result = vectorStore.store(List.of(entry));
```

### Searching Vectors

```java
VectorQuery query = new VectorQuery(
    List.of(0.15f, 0.25f, 0.35f), // Query vector
    10,                           // Top-K results
    0.1f                          // Minimum similarity score
);

Uni<List<VectorEntry>> results = vectorStore.search(query);
```

### Searching with Filters

```java
VectorQuery query = new VectorQuery(/* ... */);
Map<String, Object> filters = Map.of("category", "example");

Uni<List<VectorEntry>> results = vectorStore.search(query, filters);
```

### Deleting Vectors

```java
// Delete by IDs
Uni<Void> result = vectorStore.delete(List.of("id1", "id2"));

// Delete by filters
Map<String, Object> filters = Map.of("category", "old-data");
Uni<Void> result = vectorStore.deleteByFilters(filters);
```

## Database Schema

The module creates the following table:

```sql
CREATE TABLE wayang_vector_entries (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(DIMENSION),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

With indexes:
- Vector similarity index using HNSW algorithm
- GIN index on metadata for fast filtering

## Performance Considerations

- **Index Type**: HNSW provides better accuracy, IVFFLAT provides faster indexing
- **Vector Dimension**: Higher dimensions require more storage and computation
- **Metadata Queries**: Use GIN indexes for efficient metadata filtering
- **Connection Pooling**: Leverages PostgreSQL connection pooling

## Testing

Run the tests with:

```bash
mvn test
```

## Dependencies

- PostgreSQL 14+ with pgvector extension
- Quarkus 3.x
- Hibernate Reactive
- Vert.x Reactive PostgreSQL Client

## Migration

When upgrading, ensure the pgvector extension is installed in your PostgreSQL instance:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Security

- Connection encryption via PostgreSQL SSL support
- Role-based access control through PostgreSQL permissions
- SQL injection protection through prepared statements

## Monitoring

The implementation provides logging for:
- Query performance
- Error conditions
- Resource utilization

For more detailed monitoring, leverage PostgreSQL's built-in monitoring capabilities.