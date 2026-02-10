# Gamelan Memory Executor - Architecture Documentation

## 🏛️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Gamelan Workflow Engine                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   Control    │  │   Workflow   │  │   Executor   │                 │
│  │    Plane     │  │   Runtime    │  │   Registry   │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
│         │                  │                  │                          │
│         └──────────────────┴──────────────────┘                          │
│                            │                                             │
│                  ┌─────────┴─────────┐                                  │
│                  │  Task Scheduling  │                                  │
│                  └─────────┬─────────┘                                  │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │   gRPC/Kafka    │
                    └────────┬────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────┐
│              MEMORY-AWARE EXECUTOR LAYER                               │
├────────────────────────────┼──────────────────────────────────────────┤
│                            │                                            │
│    ┌──────────────────────▼───────────────────────┐                   │
│    │      Memory-Aware Executor Runtime            │                   │
│    │  ┌────────────┐  ┌────────────┐             │                   │
│    │  │  Executor  │  │  Executor  │ ... N more  │                   │
│    │  │     #1     │  │     #2     │             │                   │
│    │  └─────┬──────┘  └─────┬──────┘             │                   │
│    └────────┼────────────────┼────────────────────┘                   │
│             │                │                                          │
│    ┌────────┴────────────────┴────────────────────┐                   │
│    │      Context Engineering Service              │                   │
│    │  ┌──────────────────────────────────────┐   │                   │
│    │  │  - Memory Retrieval                   │   │                   │
│    │  │  - Multi-factor Scoring               │   │                   │
│    │  │  - Token Budget Optimization          │   │                   │
│    │  │  - Context Assembly                   │   │                   │
│    │  └──────────────────────────────────────┘   │                   │
│    └────────────────┬─────────────────────────────┘                   │
│                     │                                                   │
│    ┌────────────────┴─────────────────────────────┐                   │
│    │       Embedding Service                       │                   │
│    │  ┌────────┐  ┌─────────┐  ┌─────────┐      │                   │
│    │  │ OpenAI │  │ Cohere  │  │  Local  │      │                   │
│    │  └────────┘  └─────────┘  └─────────┘      │                   │
│    └────────────────┬─────────────────────────────┘                   │
│                     │                                                   │
│    ┌────────────────┴─────────────────────────────┐                   │
│    │       Vector Memory Store                     │                   │
│    │  ┌────────────────────────────────────────┐ │                   │
│    │  │  Storage Backend Abstraction           │ │                   │
│    │  └────────────────────────────────────────┘ │                   │
│    │         │           │            │           │                   │
│    │  ┌──────▼────┐ ┌───▼────┐ ┌────▼─────┐    │                   │
│    │  │ In-Memory │ │Postgres│ │ Pinecone │    │                   │
│    │  │   Store   │ │pgvector│ │ Weaviate │    │                   │
│    │  └───────────┘ └────────┘ └──────────┘    │                   │
│    └────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 Component Details

### 1. Memory-Aware Executor Runtime

**Purpose**: Manages executor lifecycle and task processing with memory awareness.

**Key Features**:
- Automatic memory storage of task executions
- Context retrieval for each task
- Adaptive importance scoring
- Multi-executor coordination

**Implementation**:
```java
@ApplicationScoped
public class ExecutorRuntime {
    - registerExecutor(WorkflowExecutor)
    - start()
    - stop()
    - handleTask(NodeExecutionTask)
}
```

### 2. Context Engineering Service

**Purpose**: Constructs optimal context from historical memories.

**Algorithm**:
```
1. Receive query/task description
2. Generate query embedding
3. Retrieve candidate memories (semantic search)
4. Multi-factor re-ranking:
   - Semantic similarity (30%)
   - Temporal recency (30%)
   - Importance score (40%)
5. Token budget optimization
6. Context assembly with sections:
   - System prompt
   - Conversation history
   - Relevant memories
   - Task instructions
7. Return engineered context
```

**Key Metrics**:
```java
class EngineerContext {
    - totalTokens: int
    - utilization: double (0.0-1.0)
    - sections: List<ContextSection>
    - relevanceScore: double
}
```

### 3. Embedding Service

**Purpose**: Convert text to vector embeddings.

**Providers**:

| Provider | Model | Dimension | Cost | Speed |
|----------|-------|-----------|------|-------|
| OpenAI | text-embedding-3-small | 1536 | $0.02/1M tokens | Fast |
| OpenAI | text-embedding-3-large | 3072 | $0.13/1M tokens | Fast |
| Cohere | embed-v3 | 1024 | $0.10/1M tokens | Fast |
| Local | TF-IDF | 384 | Free | Very Fast |

**Caching Strategy**:
```
Cache Key: SHA-256(text)
Cache Size: Configurable (default 10k entries)
Eviction: LRU
Hit Rate: ~80% in typical workloads
```

### 4. Vector Memory Store

**Purpose**: Persist and retrieve memories with vector similarity.

**Storage Options**:

#### In-Memory Store
- **Use Case**: Development, testing
- **Capacity**: Limited by RAM
- **Performance**: Fastest
- **Durability**: None

#### PostgreSQL + pgvector
- **Use Case**: Production (recommended)
- **Capacity**: Unlimited
- **Performance**: Excellent with HNSW index
- **Durability**: Full ACID guarantees

#### Pinecone/Weaviate
- **Use Case**: Cloud-native, massive scale
- **Capacity**: Unlimited
- **Performance**: Excellent
- **Durability**: Managed

## 🔄 Data Flow

### Execution Flow

```
1. Task Arrives
   ↓
2. Build Task Context
   ├─ Generate query embedding
   ├─ Search similar memories
   ├─ Re-rank by multiple factors
   └─ Assemble context within token budget
   ↓
3. Execute Task with Context
   ├─ Enhanced decision making
   └─ Context-aware processing
   ↓
4. Store Execution Memory
   ├─ Generate content embedding
   ├─ Calculate importance score
   └─ Persist to vector store
   ↓
5. Return Result
```

### Memory Lifecycle

```
[Store Memory] → [Active] → [Temporal Decay] → [Consolidation] → [Archive/Delete]
                    ↓
                [Retrieved multiple times]
                    ↓
                [Importance increases]
                    ↓
                [Promoted to Semantic Memory]
```

## 🎯 Memory Types & Use Cases

### Episodic Memory
**What**: Specific events and experiences
**Examples**:
- "Customer John Doe contacted about refund on 2024-01-15"
- "Order #12345 failed payment processing at step 3"

**Storage Duration**: 30-90 days
**Consolidation**: Yes (patterns extracted)

### Semantic Memory
**What**: Factual knowledge and patterns
**Examples**:
- "Refund policy allows 30-day returns"
- "Payment failures often due to expired cards"

**Storage Duration**: Indefinite
**Consolidation**: No (already consolidated)

### Procedural Memory
**What**: How-to knowledge and procedures
**Examples**:
- "To process refund: verify order → check policy → execute payment reversal"
- "Escalation procedure for VIP customers"

**Storage Duration**: Indefinite
**Consolidation**: Manual updates only

### Working Memory
**What**: Temporary information for current context
**Examples**:
- "Current conversation with user about shipping delay"
- "Active troubleshooting session context"

**Storage Duration**: 24 hours
**Consolidation**: No (auto-expire)

## 🔍 Search Strategies

### 1. Semantic Search
```sql
SELECT *, 1 - (embedding <=> $queryEmbedding) as similarity
FROM gamelan_memories
WHERE namespace = $namespace
ORDER BY embedding <=> $queryEmbedding
LIMIT 10
```

**Use Case**: Finding conceptually similar memories
**Performance**: ~10ms for 1M vectors with HNSW

### 2. Hybrid Search
```sql
WITH semantic AS (
  -- Vector similarity
),
keyword AS (
  -- Full-text search
)
SELECT 
  s.*,
  (s.semantic_score * 0.7 + k.keyword_score * 0.3) as combined_score
FROM semantic s
LEFT JOIN keyword k USING (id)
ORDER BY combined_score DESC
```

**Use Case**: Precision recall balance
**Performance**: ~25ms for 1M vectors

### 3. Filtered Search
```sql
SELECT *
FROM gamelan_memories
WHERE namespace = $namespace
  AND metadata @> '{"category": "refund"}'::jsonb
  AND importance >= 0.7
  AND timestamp > NOW() - INTERVAL '7 days'
ORDER BY embedding <=> $queryEmbedding
```

**Use Case**: Constrained retrieval
**Performance**: Varies with filter selectivity

## 🧮 Scoring Algorithm

### Multi-Factor Scoring Formula

```
score = (similarity × w_sim) + (recency × w_rec) + (importance × w_imp)

Where:
- similarity = cosine_similarity(query_embedding, memory_embedding)
- recency = exp(-decay_rate × age_in_minutes)
- importance = stored_importance_score
- w_sim = 0.3 (configurable)
- w_rec = 0.3 (configurable)
- w_imp = 0.4 (configurable)
```

### Importance Calculation

```java
double calculateImportance(NodeExecutionTask task, NodeExecutionResult result) {
    double base = 0.5;
    
    // Failure increases importance (learn from mistakes)
    if (result.isFailed()) base += 0.3;
    
    // Retries indicate difficulty
    if (task.attempt() > 1) base += 0.1 * min(attempt, 3);
    
    // First execution is special
    if (task.attempt() == 1) base += 0.1;
    
    // Metadata signals
    if (task.metadata.contains("critical")) base += 0.2;
    
    return min(1.0, base);
}
```

## 🔐 Security & Multi-Tenancy

### Tenant Isolation

**Namespace Format**: `{tenant}:{workflow}:{node}`

**Example**: `acme-corp:order-processing:validate`

**Enforcement Layers**:
1. Application: Namespace prefix check
2. Database: Row-level security (RLS)
3. Query: Automatic tenant filter injection

**PostgreSQL RLS Policy**:
```sql
CREATE POLICY tenant_isolation ON gamelan_memories
    FOR ALL
    TO gamelan_user
    USING (tenant_id = current_setting('app.current_tenant')::text);
```

## 📊 Performance Optimization

### Index Strategy

```sql
-- Primary indexes
CREATE INDEX idx_embedding_hnsw ON gamelan_memories 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Composite indexes
CREATE INDEX idx_tenant_namespace_time ON gamelan_memories 
    (tenant_id, namespace, timestamp DESC);

-- Partial indexes
CREATE INDEX idx_active_high_importance ON gamelan_memories (importance DESC)
    WHERE expires_at IS NULL OR expires_at > NOW();
```

### Partitioning Strategy

```sql
-- Partition by tenant for large deployments
CREATE TABLE gamelan_memories_acme PARTITION OF gamelan_memories
    FOR VALUES IN ('acme-corp');

-- Time-based partitioning for archival
CREATE TABLE gamelan_memories_2024_01 PARTITION OF gamelan_memories
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Caching Strategy

```
L1: Embedding Cache (in-memory, 10k entries)
L2: Memory Cache (Redis, 100k entries)
L3: Database (PostgreSQL with pgvector)
```

## 🚀 Scaling Considerations

### Horizontal Scaling

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Executor   │  │  Executor   │  │  Executor   │
│   Instance  │  │   Instance  │  │   Instance  │
│      #1     │  │      #2     │  │      #N     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │    PostgreSQL (Citus)         │
       │    Partitioned by tenant      │
       └───────────────────────────────┘
```

### Performance Targets

| Operation | Target Latency (p95) | Notes |
|-----------|---------------------|-------|
| Vector Search (1M) | <20ms | With HNSW index |
| Hybrid Search (1M) | <30ms | Combined semantic + keyword |
| Memory Store | <5ms | Single memory |
| Batch Store (100) | <50ms | 100 memories |
| Context Assembly | <100ms | 10 memories |
| Embedding Generation | <200ms | OpenAI API |

## 📈 Monitoring & Observability

### Key Metrics

```java
// Executor metrics
gamelan.executor.tasks.processed.total
gamelan.executor.tasks.duration.seconds
gamelan.executor.tasks.failed.total

// Memory metrics
gamelan.memory.store.operations.total
gamelan.memory.store.operations.duration
gamelan.memory.search.results.count
gamelan.memory.context.tokens.used

// Embedding metrics
gamelan.embedding.cache.hit.ratio
gamelan.embedding.api.calls.total
gamelan.embedding.api.latency
```

### Health Checks

```
/health/live   - Liveness probe
/health/ready  - Readiness probe
/metrics       - Prometheus metrics
```

## 🎓 Best Practices

### 1. Memory Management
- Set appropriate `expires_at` for working memories
- Run consolidation tasks weekly
- Monitor memory store size

### 2. Context Engineering
- Keep token budget at 70-80% of max
- Balance memory types in retrieval
- Include recent conversation history

### 3. Embedding Strategy
- Cache frequently accessed embeddings
- Batch embed when possible
- Choose model based on use case

### 4. Query Optimization
- Use metadata filters to reduce search space
- Leverage composite indexes
- Monitor slow queries

## 🔮 Future Enhancements

1. **Advanced Memory Consolidation**
   - Pattern mining from episodic memories
   - Automatic semantic memory extraction
   - Knowledge graph construction

2. **Multi-Modal Memories**
   - Image embeddings
   - Audio transcription + embedding
   - Video frame analysis

3. **Federated Learning**
   - Cross-tenant knowledge sharing (privacy-preserving)
   - Transfer learning from similar workflows

4. **Adaptive Scoring**
   - ML-based importance prediction
   - Personalized relevance scoring
   - Dynamic weight adjustment

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-09  
**Authors**: Gamelan Engineering Team



# Gamelan Memory Executor - Getting Started Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Java 21 or later
- Maven 3.9+
- Docker & Docker Compose (optional, for PostgreSQL)

### Step 1: Build the Project

```bash
cd gamelan-memory-executor
mvn clean package -DskipTests
```

### Step 2: Start with In-Memory Store (No Database Required)

```bash
# Run in dev mode with hot reload
mvn quarkus:dev
```

The application will start on port 8081.

### Step 3: Verify Installation

```bash
# Check health
curl http://localhost:8081/health/live

# Expected response:
# {"status":"UP","checks":[]}
```

### Step 4: Test the API

```bash
# Store a memory
curl -X POST http://localhost:8081/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "test",
    "content": "Customer reported damaged product and requested refund",
    "type": "EPISODIC",
    "importance": 0.8
  }'

# Search for similar memories
curl -X POST http://localhost:8081/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "test",
    "query": "product quality issue",
    "limit": 5
  }'

# Build context
curl -X POST http://localhost:8081/api/memory/context \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "test",
    "query": "How to handle refund requests?",
    "maxMemories": 3,
    "systemPrompt": "You are a support assistant"
  }'

# Get statistics
curl http://localhost:8081/api/memory/stats/test
```

## 🐘 Production Setup with PostgreSQL

### Step 1: Start PostgreSQL with pgvector

```bash
# Using Docker Compose (recommended)
docker-compose up -d postgres

# Or manually
docker run -d \
  --name gamelan-postgres \
  -e POSTGRES_DB=gamelan_memory \
  -e POSTGRES_USER=gamelan \
  -e POSTGRES_PASSWORD=gamelan \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Step 2: Initialize Database

```bash
# The database is automatically initialized on startup
# Or run manually:
docker exec -i gamelan-postgres psql -U gamelan -d gamelan_memory < init-db.sql
```

### Step 3: Configure Application

Edit `src/main/resources/application.properties`:

```properties
# Change to PostgreSQL
gamelan.memory.store.type=postgres

# Database connection (already configured)
quarkus.datasource.db-kind=postgresql
quarkus.datasource.username=gamelan
quarkus.datasource.password=gamelan
quarkus.datasource.reactive.url=postgresql://localhost:5432/gamelan_memory
```

### Step 4: Run with PostgreSQL

```bash
mvn quarkus:dev
```

## 🔑 OpenAI Integration (Optional)

### Step 1: Configure OpenAI

Edit `application.properties`:

```properties
gamelan.embedding.provider=openai
gamelan.embedding.openai.api-key=sk-your-api-key-here
gamelan.embedding.openai.model=text-embedding-3-small
```

### Step 2: Test OpenAI Embeddings

```bash
# Store memory with OpenAI embeddings
curl -X POST http://localhost:8081/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "High-quality semantic embedding test",
    "namespace": "openai-test"
  }'
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t gamelan-memory-executor:1.0.0 .
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f memory-executor

# Stop services
docker-compose down
```

## 📊 Verify Installation

Run the verification script:

```bash
#!/bin/bash

echo "=== Gamelan Memory Executor Verification ==="

# Check if service is running
if curl -sf http://localhost:8081/health/live > /dev/null; then
    echo "✅ Service is running"
else
    echo "❌ Service is not running"
    exit 1
fi

# Store test memory
STORE_RESPONSE=$(curl -s -X POST http://localhost:8081/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "verify",
    "content": "Test memory for verification",
    "importance": 0.9
  }')

if echo "$STORE_RESPONSE" | grep -q "success.*true"; then
    echo "✅ Memory storage working"
    MEMORY_ID=$(echo "$STORE_RESPONSE" | grep -o '"memoryId":"[^"]*"' | cut -d'"' -f4)
    echo "   Memory ID: $MEMORY_ID"
else
    echo "❌ Memory storage failed"
    echo "   Response: $STORE_RESPONSE"
    exit 1
fi

# Search test
SEARCH_RESPONSE=$(curl -s -X POST http://localhost:8081/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "verify",
    "query": "verification test",
    "limit": 1
  }')

if echo "$SEARCH_RESPONSE" | grep -q "success.*true"; then
    echo "✅ Memory search working"
    RESULT_COUNT=$(echo "$SEARCH_RESPONSE" | grep -o '"count":[0-9]*' | cut -d':' -f2)
    echo "   Results found: $RESULT_COUNT"
else
    echo "❌ Memory search failed"
    exit 1
fi

# Context building test
CONTEXT_RESPONSE=$(curl -s -X POST http://localhost:8081/api/memory/context \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "verify",
    "query": "Test query for context",
    "maxMemories": 3
  }')

if echo "$CONTEXT_RESPONSE" | grep -q "success.*true"; then
    echo "✅ Context engineering working"
    TOKENS=$(echo "$CONTEXT_RESPONSE" | grep -o '"totalTokens":[0-9]*' | cut -d':' -f2)
    echo "   Total tokens: $TOKENS"
else
    echo "❌ Context engineering failed"
    exit 1
fi

# Statistics test
STATS_RESPONSE=$(curl -s http://localhost:8081/api/memory/stats/verify)

if echo "$STATS_RESPONSE" | grep -q "success.*true"; then
    echo "✅ Statistics retrieval working"
    TOTAL=$(echo "$STATS_RESPONSE" | grep -o '"totalMemories":[0-9]*' | cut -d':' -f2)
    echo "   Total memories: $TOTAL"
else
    echo "❌ Statistics retrieval failed"
    exit 1
fi

echo ""
echo "=== All Verification Tests Passed ✅ ==="
echo ""
echo "Next steps:"
echo "1. Try the examples: curl http://localhost:8081/api/memory/examples/run"
echo "2. View metrics: curl http://localhost:8081/metrics"
echo "3. Check docs: cat README.md"
```

Save as `verify.sh` and run:

```bash
chmod +x verify.sh
./verify.sh
```

## 🧪 Running Tests

```bash
# Run all tests
mvn test

# Run specific test
mvn test -Dtest=MemoryExecutorTest

# Run with code coverage
mvn clean verify
```

## 📊 Monitoring

### View Metrics

```bash
# Prometheus metrics
curl http://localhost:8081/metrics

# Health checks
curl http://localhost:8081/health/live
curl http://localhost:8081/health/ready
```

### Grafana Dashboard (if monitoring profile enabled)

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Username: admin
# Password: admin
```

## 🔧 Troubleshooting

### Service won't start

```bash
# Check Java version
java -version  # Should be 21+

# Check port availability
lsof -i :8081

# View logs
docker-compose logs memory-executor
```

### PostgreSQL connection issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
docker exec gamelan-postgres psql -U gamelan -d gamelan_memory -c "SELECT 1"

# Check pgvector extension
docker exec gamelan-postgres psql -U gamelan -d gamelan_memory -c "SELECT * FROM pg_extension WHERE extname='vector'"
```

### Memory search returns empty

```bash
# Verify memories are stored
curl http://localhost:8081/api/memory/stats/your-namespace

# Check similarity threshold (lower it)
curl -X POST http://localhost:8081/api/memory/search \
  -d '{"query":"test","minSimilarity":0.0}'
```

## 📚 Next Steps

1. **Read the Documentation**
   - [README.md](README.md) - Comprehensive guide
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details

2. **Explore Examples**
   - See `src/main/java/tech/kayys/gamelan/executor/memory/examples/`
   - Run examples via API or programmatically

3. **Integrate with Workflows**
   - Use `MemoryAwareExecutor` in your workflows
   - Configure memory namespaces per workflow
   - Set up consolidation tasks

4. **Production Checklist**
   - [ ] Configure PostgreSQL with replication
   - [ ] Set up monitoring and alerts
   - [ ] Configure API keys securely
   - [ ] Enable HTTPS/TLS
   - [ ] Set up backup strategy
   - [ ] Configure resource limits
   - [ ] Set up log aggregation

## 🆘 Support

- Issues: Create a GitHub issue
- Documentation: See README.md and ARCHITECTURE.md
- Examples: Check examples directory

---

**Happy Memory-Aware Workflow Execution! 🧠🚀**