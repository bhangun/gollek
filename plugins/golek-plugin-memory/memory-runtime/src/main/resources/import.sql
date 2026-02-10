
-- Memory Sessions Table
CREATE TABLE IF NOT EXISTS memory_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP
);

-- Conversation Memories Table with Vector Support
CREATE TABLE IF NOT EXISTS conversation_memories (
    id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES memory_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT,
    timestamp TIMESTAMP NOT NULL,
    relevance_score DECIMAL(5,4),
    is_summary BOOLEAN DEFAULT FALSE,
    summary_of_session VARCHAR(255),
    embedding vector(1536)  -- Assuming OpenAI embedding dimensions
);

-- Memory Metadata Table
CREATE TABLE IF NOT EXISTS memory_metadata (
    memory_id VARCHAR(255) NOT NULL REFERENCES conversation_memories(id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    value TEXT,
    PRIMARY KEY (memory_id, key)
);

-- Session Metadata Table
CREATE TABLE IF NOT EXISTS session_metadata (
    session_id VARCHAR(255) NOT NULL REFERENCES memory_sessions(session_id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    value TEXT,
    PRIMARY KEY (session_id, key)
);

-- Execution Results Table
CREATE TABLE IF NOT EXISTS execution_results (
    id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES memory_sessions(session_id) ON DELETE CASCADE,
    request_id VARCHAR(255),
    content TEXT,
    type VARCHAR(50),
    status VARCHAR(50),
    timestamp TIMESTAMP NOT NULL
);

-- Execution Metadata Table
CREATE TABLE IF NOT EXISTS execution_metadata (
    execution_id VARCHAR(255) NOT NULL REFERENCES execution_results(id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    value TEXT,
    PRIMARY KEY (execution_id, key)
);

-- Execution Tool Calls Table
CREATE TABLE IF NOT EXISTS execution_tool_calls (
    execution_id VARCHAR(255) NOT NULL REFERENCES execution_results(id) ON DELETE CASCADE,
    tool_call TEXT NOT NULL
);

-- Memory Embeddings Table (for databases without vector support)
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id VARCHAR(255) NOT NULL REFERENCES conversation_memories(id) ON DELETE CASCADE,
    embedding_value DECIMAL(10,8) NOT NULL,
    dimension_index INTEGER NOT NULL,
    PRIMARY KEY (memory_id, dimension_index)
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_memories_session_timestamp ON conversation_memories(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_relevance ON conversation_memories(relevance_score) WHERE relevance_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_execution_session_timestamp ON execution_results(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON memory_sessions(user_id, updated_at);

-- Vector Similarity Index (if using pgvector)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON conversation_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
