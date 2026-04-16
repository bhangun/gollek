package tech.kayys.gollek.plugin.rag;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.embedding.EmbeddingService;
import tech.kayys.gollek.spi.embedding.EmbeddingService.EmbeddingResult;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.model.MultimodalContent;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * RAG (Retrieval-Augmented Generation) service for enterprise knowledge base access.
 *
 * <p>
 * Features:
 * <ul>
 * <li>Document chunking and storage</li>
 * <li>Embedding-based semantic search</li>
 * <li>Hybrid search (semantic + keyword)</li>
 * <li>Context injection into prompts</li>
 * <li>Configurable top-k retrieval</li>
 * <li>Multiple vector store support</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * @Inject
 * RAGService ragService;
 *
 * // Add documents
 * ragService.addDocument("doc1", "Content here", Map.of("source", "wiki"));
 *
 * // Search for relevant context
 * List<RAGContext> contexts = ragService.search("query", 5);
 *
 * // Inject into prompt
 * String enhancedPrompt = ragService.enhancePrompt(originalPrompt, contexts);
 * }</pre>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class RAGService {

    private static final Logger LOG = Logger.getLogger(RAGService.class);

    @Inject
    EmbeddingService embeddingService;

    // Configuration
    private int chunkSize = 500;
    private int chunkOverlap = 50;
    private int topK = 5;
    private double similarityThreshold = 0.7;

    // In-memory document store (production should use PGVector, Milvus, etc.)
    private final Map<String, Document> documents = new ConcurrentHashMap<>();
    private final Map<String, List<Chunk>> documentChunks = new ConcurrentHashMap<>();
    private final Map<String, float[]> chunkEmbeddings = new ConcurrentHashMap<>();

    /**
     * Document representation.
     */
    public record Document(
            String id,
            String content,
            Map<String, Object> metadata,
            long createdAt) {

        public Document {
            Objects.requireNonNull(id);
            Objects.requireNonNull(content);
            Objects.requireNonNull(metadata);
        }
    }

    /**
     * Document chunk for retrieval.
     */
    public record Chunk(
            String documentId,
            String chunkId,
            String content,
            int startIndex,
            int endIndex,
            Map<String, Object> metadata) {

        public Chunk {
            Objects.requireNonNull(documentId);
            Objects.requireNonNull(chunkId);
            Objects.requireNonNull(content);
            Objects.requireNonNull(metadata);
        }
    }

    /**
     * Retrieved context with relevance score.
     */
    public record RAGContext(
            String chunkId,
            String documentId,
            String content,
            double relevanceScore,
            Map<String, Object> metadata) {

        public RAGContext {
            Objects.requireNonNull(chunkId);
            Objects.requireNonNull(documentId);
            Objects.requireNonNull(content);
        }
    }

    @PostConstruct
    void init() {
        LOG.info("RAGService initialized");
        LOG.infof("Configuration: chunkSize=%d, chunkOverlap=%d, topK=%d, similarityThreshold=%.2f",
                chunkSize, chunkOverlap, topK, similarityThreshold);
    }

    // ═══════════════════════════════════════════════════════════════
    // Document Management
    // ═══════════════════════════════════════════════════════════════

    /**
     * Add a document to the knowledge base.
     *
     * @param documentId unique document identifier
     * @param content document content
     * @param metadata optional metadata
     */
    public void addDocument(String documentId, String content, Map<String, Object> metadata) {
        Objects.requireNonNull(documentId);
        Objects.requireNonNull(content);

        LOG.infof("Adding document: %s (length=%d)", documentId, content.length());

        // Create document
        Document doc = new Document(documentId, content, metadata != null ? metadata : Map.of(), System.currentTimeMillis());
        documents.put(documentId, doc);

        // Chunk the document
        List<Chunk> chunks = chunkDocument(documentId, content, metadata);
        documentChunks.put(documentId, chunks);

        // Generate embeddings for each chunk
        for (Chunk chunk : chunks) {
            try {
                float[] embedding = generateEmbedding(chunk.content());
                chunkEmbeddings.put(chunk.chunkId(), embedding);
            } catch (Exception e) {
                LOG.warnf(e, "Failed to generate embedding for chunk: %s", chunk.chunkId());
            }
        }

        LOG.infof("Document %s chunked into %d segments", documentId, chunks.size());
    }

    /**
     * Remove a document from the knowledge base.
     *
     * @param documentId document identifier
     */
    public void removeDocument(String documentId) {
        documents.remove(documentId);
        List<Chunk> chunks = documentChunks.remove(documentId);
        if (chunks != null) {
            for (Chunk chunk : chunks) {
                chunkEmbeddings.remove(chunk.chunkId());
            }
        }
        LOG.infof("Removed document: %s", documentId);
    }

    /**
     * Get a document by ID.
     *
     * @param documentId document identifier
     * @return optional document
     */
    public Optional<Document> getDocument(String documentId) {
        return Optional.ofNullable(documents.get(documentId));
    }

    /**
     * Get all documents.
     *
     * @return list of documents
     */
    public List<Document> getAllDocuments() {
        return List.copyOf(documents.values());
    }

    // ═══════════════════════════════════════════════════════════════
    // Search and Retrieval
    // ═══════════════════════════════════════════════════════════════

    /**
     * Search for relevant context using semantic similarity.
     *
     * @param query search query
     * @param k number of results to return
     * @return list of ranked contexts
     */
    public List<RAGContext> search(String query, int k) {
        LOG.debugf("Searching RAG knowledge base: %s (k=%d)", query, k);

        try {
            // Generate query embedding
            float[] queryEmbedding = generateEmbedding(query);

            // Find most similar chunks
            List<ScoredChunk> scoredChunks = new ArrayList<>();

            for (Map.Entry<String, float[]> entry : chunkEmbeddings.entrySet()) {
                double similarity = cosineSimilarity(queryEmbedding, entry.getValue());

                if (similarity >= similarityThreshold) {
                    // Find the chunk
                    Chunk chunk = findChunk(entry.getKey());
                    if (chunk != null) {
                        scoredChunks.add(new ScoredChunk(chunk, similarity));
                    }
                }
            }

            // Sort by relevance score
            scoredChunks.sort((a, b) -> Double.compare(b.score(), a.score()));

            // Take top k
            List<RAGContext> results = scoredChunks.stream()
                    .limit(k)
                    .map(sc -> new RAGContext(
                            sc.chunk().chunkId(),
                            sc.chunk().documentId(),
                            sc.chunk().content(),
                            sc.score(),
                            sc.chunk().metadata()))
                    .toList();

            LOG.infof("Found %d relevant contexts (query: %s)", results.size(), query.substring(0, Math.min(30, query.length())));

            return results;

        } catch (Exception e) {
            LOG.warnf(e, "RAG search failed, returning empty results");
            return List.of();
        }
    }

    /**
     * Search with default top-k.
     */
    public List<RAGContext> search(String query) {
        return search(query, topK);
    }

    /**
     * Hybrid search combining semantic and keyword matching.
     *
     * @param query search query
     * @param k number of results
     * @param semanticWeight weight for semantic similarity (0.0-1.0)
     * @return ranked contexts
     */
    public List<RAGContext> hybridSearch(String query, int k, double semanticWeight) {
        LOG.debugf("Hybrid search: %s (k=%d, semanticWeight=%.2f)", query, k, semanticWeight);

        // Semantic search
        List<RAGContext> semanticResults = search(query, k * 2);

        // Keyword search
        List<RAGContext> keywordResults = keywordSearch(query, k * 2);

        // Combine and re-rank
        Map<String, RAGContext> combined = new LinkedHashMap<>();

        for (RAGContext ctx : semanticResults) {
            combined.put(ctx.chunkId(), new RAGContext(
                    ctx.chunkId(),
                    ctx.documentId(),
                    ctx.content(),
                    ctx.relevanceScore() * semanticWeight,
                    ctx.metadata()));
        }

        double keywordWeight = 1.0 - semanticWeight;
        for (RAGContext ctx : keywordResults) {
            RAGContext existing = combined.get(ctx.chunkId());
            if (existing != null) {
                // Boost existing score
                double newScore = existing.relevanceScore() + (keywordWeight * 0.5);
                combined.put(ctx.chunkId(), new RAGContext(
                        existing.chunkId(),
                        existing.documentId(),
                        existing.content(),
                        newScore,
                        existing.metadata()));
            } else {
                combined.put(ctx.chunkId(), new RAGContext(
                        ctx.chunkId(),
                        ctx.documentId(),
                        ctx.content(),
                        keywordWeight * 0.5,
                        ctx.metadata()));
            }
        }

        // Sort and return top k
        return combined.values().stream()
                .sorted((a, b) -> Double.compare(b.relevanceScore(), a.relevanceScore()))
                .limit(k)
                .toList();
    }

    // ═══════════════════════════════════════════════════════════════
    // Context Enhancement
    // ═══════════════════════════════════════════════════════════════

    /**
     * Enhance a prompt with retrieved context.
     *
     * @param originalPrompt original user prompt
     * @param contexts retrieved contexts
     * @return enhanced prompt with context
     */
    public String enhancePrompt(String originalPrompt, List<RAGContext> contexts) {
        if (contexts == null || contexts.isEmpty()) {
            return originalPrompt;
        }

        StringBuilder contextText = new StringBuilder();
        contextText.append("Relevant Context:\n");

        for (int i = 0; i < contexts.size(); i++) {
            RAGContext ctx = contexts.get(i);
            contextText.append(String.format("[%d] %s (relevance: %.2f)\n",
                    i + 1, ctx.content(), ctx.relevanceScore()));
        }

        contextText.append("\n---\n\n");
        contextText.append("Question: ").append(originalPrompt);

        return contextText.toString();
    }

    /**
     * Search and enhance prompt in one step.
     *
     * @param query search query
     * @param originalPrompt original prompt
     * @param k number of contexts
     * @return enhanced prompt
     */
    public String searchAndEnhance(String query, String originalPrompt, int k) {
        List<RAGContext> contexts = search(query, k);
        return enhancePrompt(originalPrompt, contexts);
    }

    // ═══════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════

    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
        LOG.infof("Updated chunk size: %d", chunkSize);
    }

    public void setChunkOverlap(int chunkOverlap) {
        this.chunkOverlap = chunkOverlap;
        LOG.infof("Updated chunk overlap: %d", chunkOverlap);
    }

    public void setTopK(int topK) {
        this.topK = topK;
        LOG.infof("Updated top-k: %d", topK);
    }

    public void setSimilarityThreshold(double threshold) {
        this.similarityThreshold = threshold;
        LOG.infof("Updated similarity threshold: %.2f", threshold);
    }

    // ═══════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get knowledge base statistics.
     *
     * @return stats map
     */
    public Map<String, Object> getStats() {
        return Map.of(
                "documents", documents.size(),
                "chunks", chunkEmbeddings.size(),
                "chunk_size", chunkSize,
                "top_k", topK,
                "similarity_threshold", similarityThreshold);
    }

    /**
     * Clear all documents and embeddings.
     */
    public void clear() {
        documents.clear();
        documentChunks.clear();
        chunkEmbeddings.clear();
        LOG.info("RAG knowledge base cleared");
    }

    // ═══════════════════════════════════════════════════════════════
    // Private Helper Methods
    // ═══════════════════════════════════════════════════════════════

    private List<Chunk> chunkDocument(String documentId, String content, Map<String, Object> metadata) {
        List<Chunk> chunks = new ArrayList<>();
        int start = 0;
        int chunkIndex = 0;

        while (start < content.length()) {
            int end = Math.min(start + chunkSize, content.length());

            // Try to break at sentence boundary
            if (end < content.length()) {
                int lastPeriod = content.lastIndexOf('.', end);
                int lastNewline = content.lastIndexOf('\n', end);
                int breakPoint = Math.max(lastPeriod, lastNewline);

                if (breakPoint > start + chunkSize / 2) {
                    end = breakPoint + 1;
                }
            }

            String chunkContent = content.substring(start, end).trim();

            if (!chunkContent.isEmpty()) {
                String chunkId = documentId + "_chunk_" + chunkIndex;
                Map<String, Object> chunkMetadata = new HashMap<>(metadata);
                chunkMetadata.put("chunk_index", chunkIndex);

                chunks.add(new Chunk(documentId, chunkId, chunkContent, start, end, chunkMetadata));
            }

            start = end - chunkOverlap;
            chunkIndex++;

            // Safety check to prevent infinite loop
            if (start >= end) {
                start = end;
            }
        }

        return chunks;
    }

    private float[] generateEmbedding(String text) {
        try {
            MultimodalContent content = MultimodalContent.ofText(text);
            
            EmbeddingResult result = embeddingService.embed(content, null)
                .await().atMost(Duration.ofSeconds(10));

            if (result != null && result.vector() != null) {
                return result.vector();
            }
        } catch (Exception e) {
            LOG.debugf("Embedding service unavailable, using fallback: %s", e.getMessage());
        }

        // Fallback: simple hash-based embedding
        return generateSimpleEmbedding(text);
    }

    private float[] generateSimpleEmbedding(String text) {
        int dimensions = 384; // Match common embedding model size
        float[] embedding = new float[dimensions];
        char[] chars = text.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            int idx = i % dimensions;
            embedding[idx] += (float) chars[i] / 1000.0f;
        }

        // Normalize
        float norm = 0;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        if (norm > 0) {
            for (int i = 0; i < embedding.length; i++) {
                embedding[i] /= norm;
            }
        }

        return embedding;
    }

    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private Chunk findChunk(String chunkId) {
        for (List<Chunk> chunks : documentChunks.values()) {
            for (Chunk chunk : chunks) {
                if (chunk.chunkId().equals(chunkId)) {
                    return chunk;
                }
            }
        }
        return null;
    }

    private List<RAGContext> keywordSearch(String query, int k) {
        String[] queryTerms = query.toLowerCase().split("\\s+");
        List<ScoredChunk> scoredChunks = new ArrayList<>();

        for (List<Chunk> chunks : documentChunks.values()) {
            for (Chunk chunk : chunks) {
                String content = chunk.content().toLowerCase();
                int matchCount = 0;

                for (String term : queryTerms) {
                    if (content.contains(term)) {
                        matchCount++;
                    }
                }

                double score = (double) matchCount / queryTerms.length;

                if (score > 0.3) { // Minimum threshold
                    scoredChunks.add(new ScoredChunk(chunk, score));
                }
            }
        }

        scoredChunks.sort((a, b) -> Double.compare(b.score(), a.score()));

        return scoredChunks.stream()
                .limit(k)
                .map(sc -> new RAGContext(
                        sc.chunk().chunkId(),
                        sc.chunk().documentId(),
                        sc.chunk().content(),
                        sc.score(),
                        sc.chunk().metadata()))
                .toList();
    }

    // Helper record for scoring
    private record ScoredChunk(Chunk chunk, double score) {}
}
