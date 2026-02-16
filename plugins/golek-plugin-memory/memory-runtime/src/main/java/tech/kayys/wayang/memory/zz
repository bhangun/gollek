package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * ============================================================================
 * EMBEDDING SERVICE
 * ============================================================================
 * 
 * Generates vector embeddings for text using various providers:
 * - OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
 * - Cohere (embed-english-v3.0, embed-multilingual-v3.0)
 * - Sentence Transformers (local models)
 * - Custom embedding models
 * 
 * Features:
 * - Caching for frequently embedded texts
 * - Batch processing
 * - Token counting and chunking
 * - Multiple provider support with fallback
 */

// ==================== CORE INTERFACES ====================

/**
 * Embedding service interface
 */
public interface EmbeddingService {
    
    /**
     * Generate embedding for a single text
     * 
     * @param text Text to embed
     * @return Embedding vector
     */
    Uni<float[]> embed(String text);
    
    /**
     * Generate embeddings for multiple texts in batch
     * 
     * @param texts List of texts to embed
     * @return List of embedding vectors
     */
    Uni<List<float[]>> embedBatch(List<String> texts);
    
    /**
     * Get the dimension of embeddings produced by this service
     * 
     * @return Embedding dimension
     */
    int getDimension();
    
    /**
     * Get the provider name
     * 
     * @return Provider name (e.g., "openai", "cohere", "local")
     */
    String getProvider();
}

// ==================== OPENAI EMBEDDING SERVICE ====================

/**
 * OpenAI embedding service implementation
 */
@ApplicationScoped
public class OpenAIEmbeddingService implements EmbeddingService {
    
    private static final Logger LOG = LoggerFactory.getLogger(OpenAIEmbeddingService.class);
    
    @ConfigProperty(name = "gamelan.embedding.openai.api-key")
    Optional<String> apiKey;
    
    @ConfigProperty(name = "gamelan.embedding.openai.model", defaultValue = "text-embedding-3-small")
    String model;
    
    @ConfigProperty(name = "gamelan.embedding.openai.endpoint", 
                    defaultValue = "https://api.openai.com/v1/embeddings")
    String endpoint;
    
    @ConfigProperty(name = "gamelan.embedding.cache.enabled", defaultValue = "true")
    boolean cacheEnabled;
    
    @ConfigProperty(name = "gamelan.embedding.cache.max-size", defaultValue = "10000")
    int cacheMaxSize;
    
    // Cache: text hash -> embedding
    private final Map<String, float[]> embeddingCache = new ConcurrentHashMap<>();
    
    @Inject
    @RestClient
    OpenAIRestClient restClient;
    
    @Override
    public Uni<float[]> embed(String text) {
        LOG.debug("Generating embedding for text: {} chars", text.length());
        
        // Check cache
        if (cacheEnabled) {
            String textHash = hashText(text);
            float[] cached = embeddingCache.get(textHash);
            if (cached != null) {
                LOG.debug("Returning cached embedding");
                return Uni.createFrom().item(cached);
            }
        }
        
        // Prepare request
        OpenAIEmbeddingRequest request = new OpenAIEmbeddingRequest(
            model,
            text,
            "float"
        );
        
        return restClient.createEmbedding(
                "Bearer " + apiKey.orElseThrow(() -> 
                    new IllegalStateException("OpenAI API key not configured")),
                request
            )
            .map(response -> {
                if (response.data == null || response.data.isEmpty()) {
                    throw new RuntimeException("No embedding returned from OpenAI");
                }
                
                float[] embedding = response.data.get(0).embedding;
                
                // Cache the embedding
                if (cacheEnabled && embeddingCache.size() < cacheMaxSize) {
                    String textHash = hashText(text);
                    embeddingCache.put(textHash, embedding);
                }
                
                LOG.debug("Generated embedding with dimension: {}", embedding.length);
                return embedding;
            })
            .onFailure().invoke(error -> 
                LOG.error("Failed to generate embedding", error)
            );
    }
    
    @Override
    public Uni<List<float[]>> embedBatch(List<String> texts) {
        LOG.debug("Generating embeddings for batch of {} texts", texts.size());
        
        // Check cache for all texts
        List<float[]> results = new ArrayList<>();
        List<String> uncachedTexts = new ArrayList<>();
        List<Integer> uncachedIndices = new ArrayList<>();
        
        if (cacheEnabled) {
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                String textHash = hashText(text);
                float[] cached = embeddingCache.get(textHash);
                
                if (cached != null) {
                    results.add(cached);
                } else {
                    results.add(null); // Placeholder
                    uncachedTexts.add(text);
                    uncachedIndices.add(i);
                }
            }
            
            if (uncachedTexts.isEmpty()) {
                LOG.debug("All embeddings found in cache");
                return Uni.createFrom().item(results);
            }
        } else {
            uncachedTexts.addAll(texts);
            for (int i = 0; i < texts.size(); i++) {
                results.add(null);
                uncachedIndices.add(i);
            }
        }
        
        // Batch request for uncached texts
        OpenAIEmbeddingBatchRequest request = new OpenAIEmbeddingBatchRequest(
            model,
            uncachedTexts,
            "float"
        );
        
        return restClient.createEmbeddingBatch(
                "Bearer " + apiKey.orElseThrow(() -> 
                    new IllegalStateException("OpenAI API key not configured")),
                request
            )
            .map(response -> {
                if (response.data == null || response.data.size() != uncachedTexts.size()) {
                    throw new RuntimeException("Invalid batch embedding response");
                }
                
                // Fill in the results and cache
                for (int i = 0; i < uncachedTexts.size(); i++) {
                    float[] embedding = response.data.get(i).embedding;
                    int originalIndex = uncachedIndices.get(i);
                    results.set(originalIndex, embedding);
                    
                    // Cache
                    if (cacheEnabled && embeddingCache.size() < cacheMaxSize) {
                        String textHash = hashText(uncachedTexts.get(i));
                        embeddingCache.put(textHash, embedding);
                    }
                }
                
                LOG.debug("Generated {} embeddings", response.data.size());
                return results;
            });
    }
    
    @Override
    public int getDimension() {
        return switch (model) {
            case "text-embedding-3-small" -> 1536;
            case "text-embedding-3-large" -> 3072;
            case "text-embedding-ada-002" -> 1536;
            default -> 1536;
        };
    }
    
    @Override
    public String getProvider() {
        return "openai";
    }
    
    /**
     * Hash text for cache key
     */
    private String hashText(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(text.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(hash);
        } catch (Exception e) {
            return text; // Fallback
        }
    }
}

// ==================== LOCAL EMBEDDING SERVICE ====================

/**
 * Local embedding service using simple TF-IDF approach
 * For development and testing when external APIs are not available
 */
@ApplicationScoped
public class LocalTFIDFEmbeddingService implements EmbeddingService {
    
    private static final Logger LOG = LoggerFactory.getLogger(LocalTFIDFEmbeddingService.class);
    private static final int DIMENSION = 384; // Common for local models
    private static final Pattern WORD_PATTERN = Pattern.compile("\\w+");
    
    // Document frequency: word -> count
    private final Map<String, Integer> documentFrequency = new ConcurrentHashMap<>();
    private final Set<String> vocabulary = ConcurrentHashMap.newKeySet();
    private int totalDocuments = 0;
    
    @Override
    public Uni<float[]> embed(String text) {
        LOG.debug("Generating local TF-IDF embedding for: {} chars", text.length());
        
        return Uni.createFrom().item(() -> {
            // Tokenize and calculate term frequency
            Map<String, Integer> termFrequency = calculateTermFrequency(text);
            
            // Update global statistics
            updateDocumentStatistics(termFrequency.keySet());
            
            // Generate TF-IDF vector
            float[] embedding = generateTFIDFVector(termFrequency);
            
            // Normalize
            normalize(embedding);
            
            return embedding;
        });
    }
    
    @Override
    public Uni<List<float[]>> embedBatch(List<String> texts) {
        LOG.debug("Generating local embeddings for batch of {} texts", texts.size());
        
        return Uni.createFrom().item(() -> {
            List<float[]> embeddings = new ArrayList<>();
            
            for (String text : texts) {
                Map<String, Integer> termFrequency = calculateTermFrequency(text);
                updateDocumentStatistics(termFrequency.keySet());
                float[] embedding = generateTFIDFVector(termFrequency);
                normalize(embedding);
                embeddings.add(embedding);
            }
            
            return embeddings;
        });
    }
    
    @Override
    public int getDimension() {
        return DIMENSION;
    }
    
    @Override
    public String getProvider() {
        return "local-tfidf";
    }
    
    /**
     * Calculate term frequency for text
     */
    private Map<String, Integer> calculateTermFrequency(String text) {
        Map<String, Integer> tf = new HashMap<>();
        
        var matcher = WORD_PATTERN.matcher(text.toLowerCase());
        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2) { // Skip short words
                tf.merge(word, 1, Integer::sum);
            }
        }
        
        return tf;
    }
    
    /**
     * Update global document statistics
     */
    private synchronized void updateDocumentStatistics(Set<String> words) {
        totalDocuments++;
        
        for (String word : words) {
            vocabulary.add(word);
            documentFrequency.merge(word, 1, Integer::sum);
        }
    }
    
    /**
     * Generate TF-IDF vector
     */
    private float[] generateTFIDFVector(Map<String, Integer> termFrequency) {
        float[] vector = new float[DIMENSION];
        
        // Convert vocabulary to indexed list for consistent hashing
        List<String> vocabList = new ArrayList<>(vocabulary);
        Collections.sort(vocabList);
        
        for (Map.Entry<String, Integer> entry : termFrequency.entrySet()) {
            String term = entry.getKey();
            int tf = entry.getValue();
            
            // Calculate IDF
            int df = documentFrequency.getOrDefault(term, 1);
            double idf = Math.log((double) (totalDocuments + 1) / (df + 1));
            
            // Calculate TF-IDF
            double tfidf = tf * idf;
            
            // Hash term to dimension
            int index = Math.abs(term.hashCode()) % DIMENSION;
            vector[index] += (float) tfidf;
        }
        
        return vector;
    }
    
    /**
     * Normalize vector to unit length
     */
    private void normalize(float[] vector) {
        double sum = 0.0;
        for (float v : vector) {
            sum += v * v;
        }
        
        if (sum > 0) {
            double norm = Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}

// ==================== EMBEDDING SERVICE FACTORY ====================

/**
 * Factory for creating embedding service based on configuration
 */
@ApplicationScoped
public class EmbeddingServiceFactory {
    
    private static final Logger LOG = LoggerFactory.getLogger(EmbeddingServiceFactory.class);
    
    @ConfigProperty(name = "gamelan.embedding.provider", defaultValue = "local")
    String provider;
    
    @Inject
    OpenAIEmbeddingService openAIService;
    
    @Inject
    LocalTFIDFEmbeddingService localService;
    
    /**
     * Get the configured embedding service
     */
    public EmbeddingService getEmbeddingService() {
        LOG.info("Using embedding provider: {}", provider);
        
        return switch (provider.toLowerCase()) {
            case "openai" -> openAIService;
            case "local", "local-tfidf" -> localService;
            default -> {
                LOG.warn("Unknown provider: {}, falling back to local", provider);
                yield localService;
            }
        };
    }
}

// ==================== REST CLIENT MODELS ====================

/**
 * OpenAI REST client interface
 */
@org.eclipse.microprofile.rest.client.inject.RegisterRestClient(configKey = "openai")
@jakarta.ws.rs.Path("/v1")
public interface OpenAIRestClient {
    
    @jakarta.ws.rs.POST
    @jakarta.ws.rs.Path("/embeddings")
    @jakarta.ws.rs.Produces(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    @jakarta.ws.rs.Consumes(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    Uni<OpenAIEmbeddingResponse> createEmbedding(
        @jakarta.ws.rs.HeaderParam("Authorization") String authorization,
        OpenAIEmbeddingRequest request
    );
    
    @jakarta.ws.rs.POST
    @jakarta.ws.rs.Path("/embeddings")
    @jakarta.ws.rs.Produces(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    @jakarta.ws.rs.Consumes(jakarta.ws.rs.core.MediaType.APPLICATION_JSON)
    Uni<OpenAIEmbeddingResponse> createEmbeddingBatch(
        @jakarta.ws.rs.HeaderParam("Authorization") String authorization,
        OpenAIEmbeddingBatchRequest request
    );
}

/**
 * OpenAI embedding request (single)
 */
class OpenAIEmbeddingRequest {
    public final String model;
    public final String input;
    public final String encoding_format;
    
    public OpenAIEmbeddingRequest(String model, String input, String encoding_format) {
        this.model = model;
        this.input = input;
        this.encoding_format = encoding_format;
    }
}

/**
 * OpenAI embedding request (batch)
 */
class OpenAIEmbeddingBatchRequest {
    public final String model;
    public final List<String> input;
    public final String encoding_format;
    
    public OpenAIEmbeddingBatchRequest(String model, List<String> input, String encoding_format) {
        this.model = model;
        this.input = input;
        this.encoding_format = encoding_format;
    }
}

/**
 * OpenAI embedding response
 */
class OpenAIEmbeddingResponse {
    public String object;
    public List<EmbeddingData> data;
    public String model;
    public Usage usage;
    
    static class EmbeddingData {
        public String object;
        public float[] embedding;
        public int index;
    }
    
    static class Usage {
        public int prompt_tokens;
        public int total_tokens;
    }
}

// ==================== TEXT CHUNKING UTILITIES ====================

/**
 * Utilities for chunking text into smaller pieces for embedding
 */
@ApplicationScoped
public class TextChunker {
    
    private static final Logger LOG = LoggerFactory.getLogger(TextChunker.class);
    
    @ConfigProperty(name = "gamelan.embedding.chunk.size", defaultValue = "512")
    int chunkSize;
    
    @ConfigProperty(name = "gamelan.embedding.chunk.overlap", defaultValue = "50")
    int chunkOverlap;
    
    /**
     * Split text into chunks with overlap
     * 
     * @param text Text to chunk
     * @return List of text chunks
     */
    public List<String> chunk(String text) {
        List<String> chunks = new ArrayList<>();
        
        if (text == null || text.isEmpty()) {
            return chunks;
        }
        
        // Split by sentences first
        String[] sentences = text.split("(?<=[.!?])\\s+");
        
        StringBuilder currentChunk = new StringBuilder();
        int currentLength = 0;
        
        for (String sentence : sentences) {
            int sentenceLength = sentence.length();
            
            if (currentLength + sentenceLength > chunkSize && currentLength > 0) {
                // Save current chunk
                chunks.add(currentChunk.toString().trim());
                
                // Start new chunk with overlap
                currentChunk = new StringBuilder();
                currentLength = 0;
                
                // Add overlap from previous chunk
                String prevChunk = chunks.get(chunks.size() - 1);
                if (prevChunk.length() > chunkOverlap) {
                    String overlap = prevChunk.substring(prevChunk.length() - chunkOverlap);
                    currentChunk.append(overlap).append(" ");
                    currentLength = overlap.length();
                }
            }
            
            currentChunk.append(sentence).append(" ");
            currentLength += sentenceLength;
        }
        
        // Add final chunk
        if (currentLength > 0) {
            chunks.add(currentChunk.toString().trim());
        }
        
        LOG.debug("Chunked text of {} chars into {} chunks", text.length(), chunks.size());
        
        return chunks;
    }
    
    /**
     * Estimate token count (rough approximation)
     * 
     * @param text Text to count
     * @return Estimated token count
     */
    public int estimateTokenCount(String text) {
        // Rough approximation: 1 token ≈ 4 characters
        return text.length() / 4;
    }
}

package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * ============================================================================
 * CONTEXT ENGINEERING SERVICE
 * ============================================================================
 * 
 * Advanced context engineering and prompt optimization for AI agents.
 * 
 * Features:
 * - Dynamic context window management
 * - Relevance-based memory retrieval
 * - Temporal decay and recency bias
 * - Token budget optimization
 * - Multi-stage context assembly
 * - Prompt template management
 * - Context compression
 */

// ==================== CONTEXT ENGINEERING SERVICE ====================

@ApplicationScoped
public class ContextEngineeringService {
    
    private static final Logger LOG = LoggerFactory.getLogger(ContextEngineeringService.class);
    
    @ConfigProperty(name = "gamelan.context.max-tokens", defaultValue = "8000")
    int maxContextTokens;
    
    @ConfigProperty(name = "gamelan.context.decay-rate", defaultValue = "0.001")
    double temporalDecayRate;
    
    @ConfigProperty(name = "gamelan.context.recency-weight", defaultValue = "0.3")
    double recencyWeight;
    
    @ConfigProperty(name = "gamelan.context.importance-weight", defaultValue = "0.4")
    double importanceWeight;
    
    @ConfigProperty(name = "gamelan.context.similarity-weight", defaultValue = "0.3")
    double similarityWeight;
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    TextChunker textChunker;
    
    /**
     * Build optimized context from memories and current query
     * 
     * @param query Current query/instruction
     * @param namespace Memory namespace
     * @param config Context configuration
     * @return Optimized context object
     */
    public Uni<EngineerContext> buildContext(
            String query,
            String namespace,
            ContextConfig config) {
        
        LOG.info("Building context for query in namespace: {}", namespace);
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // Generate query embedding
        return embeddingService.embed(query)
            .flatMap(queryEmbedding -> 
                retrieveRelevantMemories(
                    queryEmbedding,
                    namespace,
                    config
                )
            )
            .map(memories -> 
                assembleContext(query, memories, config)
            );
    }
    
    /**
     * Retrieve relevant memories with multi-factor scoring
     * 
     * @param queryEmbedding Query vector embedding
     * @param namespace Memory namespace
     * @param config Context configuration
     * @return List of relevant memories
     */
    private Uni<List<ScoredMemory>> retrieveRelevantMemories(
            float[] queryEmbedding,
            String namespace,
            ContextConfig config) {
        
        LOG.debug("Retrieving relevant memories with multi-factor scoring");
        
        // Search for semantically similar memories
        Map<String, Object> filters = new HashMap<>();
        filters.put("namespace", namespace);
        
        // Add type filters if specified
        if (config.getMemoryTypes() != null && !config.getMemoryTypes().isEmpty()) {
            filters.put("types", config.getMemoryTypes());
        }
        
        return memoryStore.search(
                queryEmbedding,
                config.getMaxMemories() * 2, // Retrieve more, then re-rank
                0.5, // Minimum similarity
                filters
            )
            .map(scoredMemories -> 
                reRankWithMultipleFactors(scoredMemories, queryEmbedding)
            )
            .map(reranked -> 
                reranked.stream()
                    .limit(config.getMaxMemories())
                    .collect(Collectors.toList())
            );
    }
    
    /**
     * Re-rank memories using multiple factors:
     * - Semantic similarity
     * - Temporal recency
     * - Importance score
     * - Access frequency
     */
    private List<ScoredMemory> reRankWithMultipleFactors(
            List<ScoredMemory> memories,
            float[] queryEmbedding) {
        
        LOG.debug("Re-ranking {} memories with multiple factors", memories.size());
        
        List<ScoredMemory> reranked = new ArrayList<>();
        Instant now = Instant.now();
        
        for (ScoredMemory scoredMemory : memories) {
            Memory memory = scoredMemory.getMemory();
            
            // Original similarity score
            double similarityScore = scoredMemory.getScore();
            
            // Recency score (exponential decay)
            long ageMinutes = Duration.between(memory.getTimestamp(), now).toMinutes();
            double recencyScore = Math.exp(-temporalDecayRate * ageMinutes);
            
            // Importance score (from memory)
            double importanceScore = memory.getImportance();
            
            // Access frequency (from metadata)
            int accessCount = (int) memory.getMetadata().getOrDefault("accessCount", 0);
            double frequencyScore = Math.log(accessCount + 1) / Math.log(100); // Normalized log
            
            // Combined score
            double combinedScore = 
                (similarityScore * similarityWeight) +
                (recencyScore * recencyWeight) +
                (importanceScore * importanceWeight) +
                (frequencyScore * 0.1); // Small weight for frequency
            
            Map<String, Object> scoreBreakdown = Map.of(
                "total", combinedScore,
                "similarity", similarityScore,
                "recency", recencyScore,
                "importance", importanceScore,
                "frequency", frequencyScore
            );
            
            reranked.add(new ScoredMemory(memory, combinedScore, scoreBreakdown));
        }
        
        // Sort by combined score
        reranked.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));
        
        LOG.debug("Re-ranking complete, top score: {}", 
            reranked.isEmpty() ? 0 : reranked.get(0).getScore());
        
        return reranked;
    }
    
    /**
     * Assemble context from memories with token budget optimization
     * 
     * @param query Original query
     * @param memories Relevant memories
     * @param config Context configuration
     * @return Assembled context
     */
    private EngineerContext assembleContext(
            String query,
            List<ScoredMemory> memories,
            ContextConfig config) {
        
        LOG.debug("Assembling context from {} memories", memories.size());
        
        // Calculate token budget
        int queryTokens = textChunker.estimateTokenCount(query);
        int availableTokens = maxContextTokens - queryTokens - config.getReservedTokens();
        
        // Build context sections
        List<ContextSection> sections = new ArrayList<>();
        int usedTokens = 0;
        
        // 1. System instructions (if provided)
        if (config.getSystemPrompt() != null) {
            int systemTokens = textChunker.estimateTokenCount(config.getSystemPrompt());
            if (usedTokens + systemTokens <= availableTokens) {
                sections.add(new ContextSection(
                    "system",
                    config.getSystemPrompt(),
                    systemTokens,
                    1.0
                ));
                usedTokens += systemTokens;
            }
        }
        
        // 2. Recent conversation history (if provided)
        if (config.getConversationHistory() != null) {
            int historyTokens = textChunker.estimateTokenCount(
                String.join("\n", config.getConversationHistory()));
            if (usedTokens + historyTokens <= availableTokens * 0.3) { // Max 30% for history
                sections.add(new ContextSection(
                    "conversation_history",
                    String.join("\n", config.getConversationHistory()),
                    historyTokens,
                    0.9
                ));
                usedTokens += historyTokens;
            }
        }
        
        // 3. Relevant memories
        for (ScoredMemory scoredMemory : memories) {
            Memory memory = scoredMemory.getMemory();
            String content = formatMemory(memory, config);
            int memoryTokens = textChunker.estimateTokenCount(content);
            
            if (usedTokens + memoryTokens <= availableTokens) {
                sections.add(new ContextSection(
                    "memory_" + memory.getType().name().toLowerCase(),
                    content,
                    memoryTokens,
                    scoredMemory.getScore()
                ));
                usedTokens += memoryTokens;
                
                // Update access count
                Map<String, Object> metadata = new HashMap<>(memory.getMetadata());
                int accessCount = (int) metadata.getOrDefault("accessCount", 0);
                metadata.put("accessCount", accessCount + 1);
                metadata.put("lastAccessed", Instant.now().toString());
                
                memoryStore.updateMetadata(memory.getId(), metadata)
                    .subscribe().with(
                        updated -> LOG.trace("Updated access count for memory: {}", memory.getId()),
                        error -> LOG.warn("Failed to update memory metadata", error)
                    );
            } else {
                LOG.debug("Token budget exhausted, skipping remaining memories");
                break;
            }
        }
        
        // 4. Task instructions (if provided)
        if (config.getTaskInstructions() != null) {
            int instructionTokens = textChunker.estimateTokenCount(config.getTaskInstructions());
            if (usedTokens + instructionTokens <= availableTokens) {
                sections.add(new ContextSection(
                    "task_instructions",
                    config.getTaskInstructions(),
                    instructionTokens,
                    1.0
                ));
                usedTokens += instructionTokens;
            }
        }
        
        EngineerContext context = new EngineerContext(
            query,
            sections,
            usedTokens,
            maxContextTokens
        );
        
        LOG.info("Context assembled: {} sections, {} tokens used of {} available",
            sections.size(), usedTokens, availableTokens);
        
        return context;
    }
    
    /**
     * Format memory for inclusion in context
     * 
     * @param memory Memory to format
     * @param config Context configuration
     * @return Formatted memory string
     */
    private String formatMemory(Memory memory, ContextConfig config) {
        StringBuilder formatted = new StringBuilder();
        
        // Add metadata if enabled
        if (config.isIncludeMetadata()) {
            formatted.append("[")
                .append(memory.getType().name())
                .append(" - ")
                .append(formatTimestamp(memory.getTimestamp()))
                .append(" - Importance: ")
                .append(String.format("%.2f", memory.getImportance()))
                .append("]\n");
        }
        
        // Add content
        formatted.append(memory.getContent());
        
        return formatted.toString();
    }
    
    /**
     * Format timestamp for human readability
     */
    private String formatTimestamp(Instant timestamp) {
        Duration age = Duration.between(timestamp, Instant.now());
        
        if (age.toDays() > 0) {
            return age.toDays() + " days ago";
        } else if (age.toHours() > 0) {
            return age.toHours() + " hours ago";
        } else if (age.toMinutes() > 0) {
            return age.toMinutes() + " minutes ago";
        } else {
            return "just now";
        }
    }
}

// ==================== CONFIGURATION MODELS ====================

/**
 * Context configuration
 */
public class ContextConfig {
    
    private int maxMemories = 10;
    private int reservedTokens = 1000; // Reserve for response
    private List<MemoryType> memoryTypes;
    private String systemPrompt;
    private List<String> conversationHistory;
    private String taskInstructions;
    private boolean includeMetadata = true;
    
    // Getters and setters
    public int getMaxMemories() { return maxMemories; }
    public void setMaxMemories(int maxMemories) { this.maxMemories = maxMemories; }
    
    public int getReservedTokens() { return reservedTokens; }
    public void setReservedTokens(int reservedTokens) { this.reservedTokens = reservedTokens; }
    
    public List<MemoryType> getMemoryTypes() { return memoryTypes; }
    public void setMemoryTypes(List<MemoryType> memoryTypes) { this.memoryTypes = memoryTypes; }
    
    public String getSystemPrompt() { return systemPrompt; }
    public void setSystemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; }
    
    public List<String> getConversationHistory() { return conversationHistory; }
    public void setConversationHistory(List<String> conversationHistory) { 
        this.conversationHistory = conversationHistory; 
    }
    
    public String getTaskInstructions() { return taskInstructions; }
    public void setTaskInstructions(String taskInstructions) { 
        this.taskInstructions = taskInstructions; 
    }
    
    public boolean isIncludeMetadata() { return includeMetadata; }
    public void setIncludeMetadata(boolean includeMetadata) { 
        this.includeMetadata = includeMetadata; 
    }
    
    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private final ContextConfig config = new ContextConfig();
        
        public Builder maxMemories(int maxMemories) {
            config.maxMemories = maxMemories;
            return this;
        }
        
        public Builder reservedTokens(int reservedTokens) {
            config.reservedTokens = reservedTokens;
            return this;
        }
        
        public Builder memoryTypes(List<MemoryType> types) {
            config.memoryTypes = types;
            return this;
        }
        
        public Builder systemPrompt(String prompt) {
            config.systemPrompt = prompt;
            return this;
        }
        
        public Builder conversationHistory(List<String> history) {
            config.conversationHistory = history;
            return this;
        }
        
        public Builder taskInstructions(String instructions) {
            config.taskInstructions = instructions;
            return this;
        }
        
        public Builder includeMetadata(boolean include) {
            config.includeMetadata = include;
            return this;
        }
        
        public ContextConfig build() {
            return config;
        }
    }
}

/**
 * Context section
 */
public class ContextSection {
    
    private final String type;
    private final String content;
    private final int tokenCount;
    private final double relevanceScore;
    
    public ContextSection(String type, String content, int tokenCount, double relevanceScore) {
        this.type = type;
        this.content = content;
        this.tokenCount = tokenCount;
        this.relevanceScore = relevanceScore;
    }
    
    public String getType() { return type; }
    public String getContent() { return content; }
    public int getTokenCount() { return tokenCount; }
    public double getRelevanceScore() { return relevanceScore; }
}

/**
 * Engineered context ready for LLM consumption
 */
public class EngineerContext {
    
    private final String query;
    private final List<ContextSection> sections;
    private final int totalTokens;
    private final int maxTokens;
    
    public EngineerContext(
            String query,
            List<ContextSection> sections,
            int totalTokens,
            int maxTokens) {
        this.query = query;
        this.sections = sections;
        this.totalTokens = totalTokens;
        this.maxTokens = maxTokens;
    }
    
    /**
     * Get formatted prompt ready for LLM
     */
    public String toPrompt() {
        StringBuilder prompt = new StringBuilder();
        
        // Add sections in order
        for (ContextSection section : sections) {
            if (section.getType().startsWith("memory_")) {
                prompt.append("\n--- Relevant Context ---\n");
            } else if (section.getType().equals("conversation_history")) {
                prompt.append("\n--- Recent Conversation ---\n");
            } else if (section.getType().equals("task_instructions")) {
                prompt.append("\n--- Task ---\n");
            }
            
            prompt.append(section.getContent());
            prompt.append("\n");
        }
        
        // Add query at the end
        prompt.append("\n--- Current Query ---\n");
        prompt.append(query);
        
        return prompt.toString();
    }
    
    /**
     * Get structured representation for advanced LLMs
     */
    public Map<String, Object> toStructured() {
        Map<String, Object> structured = new HashMap<>();
        
        // Group sections by type
        for (ContextSection section : sections) {
            String type = section.getType();
            
            if (type.startsWith("memory_")) {
                List<String> memories = (List<String>) structured.computeIfAbsent(
                    "memories", k -> new ArrayList<String>());
                memories.add(section.getContent());
            } else {
                structured.put(type, section.getContent());
            }
        }
        
        structured.put("query", query);
        structured.put("metadata", Map.of(
            "totalTokens", totalTokens,
            "maxTokens", maxTokens,
            "sectionCount", sections.size()
        ));
        
        return structured;
    }
    
    // Getters
    public String getQuery() { return query; }
    public List<ContextSection> getSections() { return sections; }
    public int getTotalTokens() { return totalTokens; }
    public int getMaxTokens() { return maxTokens; }
    public double getUtilization() { return (double) totalTokens / maxTokens; }
}

// ==================== PROMPT TEMPLATE MANAGER ====================

/**
 * Manages prompt templates for different use cases
 */
@ApplicationScoped
public class PromptTemplateManager {
    
    private static final Logger LOG = LoggerFactory.getLogger(PromptTemplateManager.class);
    
    private final Map<String, PromptTemplate> templates = new HashMap<>();
    
    public PromptTemplateManager() {
        initializeDefaultTemplates();
    }
    
    /**
     * Initialize default templates
     */
    private void initializeDefaultTemplates() {
        // Question answering template
        templates.put("qa", new PromptTemplate(
            "qa",
            """
            You are a helpful AI assistant. Use the following context to answer the user's question.
            If you cannot find the answer in the context, say so honestly.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """,
            List.of("context", "query")
        ));
        
        // Task execution template
        templates.put("task", new PromptTemplate(
            "task",
            """
            You are an AI agent capable of executing tasks. Based on the context and your capabilities,
            execute the following task.
            
            Available Context:
            {context}
            
            Task: {query}
            
            Please provide your response in the following format:
            1. Analysis: Brief analysis of the task
            2. Action: The action you will take
            3. Result: Expected or actual result
            """,
            List.of("context", "query")
        ));
        
        // Conversational template
        templates.put("chat", new PromptTemplate(
            "chat",
            """
            You are a friendly AI assistant having a conversation with a user.
            Use the conversation history and any relevant context to provide helpful responses.
            
            {conversation_history}
            
            Relevant Context:
            {context}
            
            User: {query}
            
            Assistant:
            """,
            List.of("conversation_history", "context", "query")
        ));
        
        LOG.info("Initialized {} prompt templates", templates.size());
    }
    
    /**
     * Get template by name
     */
    public PromptTemplate getTemplate(String name) {
        return templates.get(name);
    }
    
    /**
     * Register custom template
     */
    public void registerTemplate(PromptTemplate template) {
        templates.put(template.getName(), template);
        LOG.info("Registered custom template: {}", template.getName());
    }
    
    /**
     * Apply template with variables
     */
    public String applyTemplate(String templateName, Map<String, String> variables) {
        PromptTemplate template = templates.get(templateName);
        if (template == null) {
            throw new IllegalArgumentException("Template not found: " + templateName);
        }
        
        return template.apply(variables);
    }
}

/**
 * Prompt template
 */
public class PromptTemplate {
    
    private final String name;
    private final String template;
    private final List<String> requiredVariables;
    
    public PromptTemplate(String name, String template, List<String> requiredVariables) {
        this.name = name;
        this.template = template;
        this.requiredVariables = requiredVariables;
    }
    
    /**
     * Apply template with variables
     */
    public String apply(Map<String, String> variables) {
        String result = template;
        
        // Validate required variables
        for (String required : requiredVariables) {
            if (!variables.containsKey(required)) {
                throw new IllegalArgumentException("Missing required variable: " + required);
            }
        }
        
        // Replace variables
        for (Map.Entry<String, String> entry : variables.entrySet()) {
            String placeholder = "{" + entry.getKey() + "}";
            result = result.replace(placeholder, entry.getValue());
        }
        
        return result;
    }
    
    public String getName() { return name; }
    public String getTemplate() { return template; }
    public List<String> getRequiredVariables() { return requiredVariables; }
}

package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gamelan.core.domain.*;
import tech.kayys.gamelan.core.engine.NodeExecutionResult;
import tech.kayys.gamelan.core.engine.NodeExecutionTask;
import tech.kayys.gamelan.executor.AbstractWorkflowExecutor;
import tech.kayys.gamelan.executor.Executor;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * ============================================================================
 * MEMORY-AWARE WORKFLOW EXECUTOR
 * ============================================================================
 * 
 * Workflow executor that leverages semantic memory and context engineering
 * to enhance task execution with relevant historical context.
 * 
 * Features:
 * - Automatic memory storage of task executions
 * - Context-aware task execution
 * - Learning from past executions
 * - Adaptive importance scoring
 * - Memory consolidation and cleanup
 */

@Executor(
    executorType = "memory-aware-executor",
    communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC,
    maxConcurrentTasks = 10
)
@ApplicationScoped
public class MemoryAwareExecutor extends AbstractWorkflowExecutor {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryAwareExecutor.class);
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    ContextEngineeringService contextService;
    
    @Inject
    TextChunker textChunker;
    
    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Executing memory-aware task: run={}, node={}",
            task.runId().value(), task.nodeId().value());
        
        String namespace = buildNamespace(task);
        String taskDescription = buildTaskDescription(task);
        
        // Build context from memory
        ContextConfig contextConfig = ContextConfig.builder()
            .maxMemories(5)
            .memoryTypes(List.of(MemoryType.EPISODIC, MemoryType.SEMANTIC))
            .includeMetadata(true)
            .build();
        
        return contextService.buildContext(taskDescription, namespace, contextConfig)
            .flatMap(context -> executeWithContext(task, context))
            .flatMap(result -> storeExecutionMemory(task, result, namespace)
                .replaceWith(result))
            .onFailure().invoke(error -> 
                LOG.error("Memory-aware execution failed", error));
    }
    
    /**
     * Execute task with engineered context
     */
    private Uni<NodeExecutionResult> executeWithContext(
            NodeExecutionTask task,
            EngineerContext context) {
        
        LOG.debug("Executing with context: {} tokens, {} sections",
            context.getTotalTokens(), context.getSections().size());
        
        // Extract task-specific logic from context
        Map<String, Object> taskContext = task.context();
        
        // Simulate task execution with memory context
        // In real implementation, this would call actual task logic
        return Uni.createFrom().item(() -> {
            Map<String, Object> output = new HashMap<>(taskContext);
            
            // Add context insights
            output.put("contextSections", context.getSections().size());
            output.put("contextTokens", context.getTotalTokens());
            output.put("contextUtilization", context.getUtilization());
            
            // Add relevant memories summary
            List<String> relevantInsights = context.getSections().stream()
                .filter(s -> s.getType().startsWith("memory_"))
                .map(s -> s.getContent().substring(0, Math.min(100, s.getContent().length())))
                .toList();
            output.put("relevantInsights", relevantInsights);
            
            return NodeExecutionResult.success(
                task.runId(),
                task.nodeId(),
                task.attempt(),
                output,
                task.token()
            );
        });
    }
    
    /**
     * Store execution as memory for future reference
     */
    private Uni<String> storeExecutionMemory(
            NodeExecutionTask task,
            NodeExecutionResult result,
            String namespace) {
        
        LOG.debug("Storing execution memory for task: {}", task.nodeId().value());
        
        // Build memory content
        String memoryContent = buildMemoryContent(task, result);
        
        // Calculate importance based on execution characteristics
        double importance = calculateImportance(task, result);
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // Generate embedding and store
        return embeddingService.embed(memoryContent)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace(namespace)
                    .content(memoryContent)
                    .embedding(embedding)
                    .type(MemoryType.EPISODIC)
                    .importance(importance)
                    .timestamp(Instant.now())
                    .addMetadata("runId", task.runId().value())
                    .addMetadata("nodeId", task.nodeId().value())
                    .addMetadata("attempt", task.attempt())
                    .addMetadata("status", result.status().name())
                    .addMetadata("executionTime", System.currentTimeMillis())
                    .addMetadata("accessCount", 0)
                    .build();
                
                return memoryStore.store(memory);
            });
    }
    
    /**
     * Build namespace for memory isolation
     */
    private String buildNamespace(NodeExecutionTask task) {
        // Namespace format: workflowDef:nodeType
        Map<String, Object> context = task.context();
        String workflowDef = (String) context.getOrDefault("workflowDefinitionId", "default");
        return "workflow:" + workflowDef;
    }
    
    /**
     * Build task description for context retrieval
     */
    private String buildTaskDescription(NodeExecutionTask task) {
        StringBuilder description = new StringBuilder();
        
        Map<String, Object> context = task.context();
        
        description.append("Task: ").append(task.nodeId().value()).append("\n");
        description.append("Execution context:\n");
        
        // Add key context variables
        context.forEach((key, value) -> {
            if (value != null && !key.equals("internalState")) {
                description.append("- ").append(key).append(": ")
                    .append(value.toString().substring(0, Math.min(100, value.toString().length())))
                    .append("\n");
            }
        });
        
        return description.toString();
    }
    
    /**
     * Build memory content from execution
     */
    private String buildMemoryContent(NodeExecutionTask task, NodeExecutionResult result) {
        StringBuilder content = new StringBuilder();
        
        content.append("Execution Summary\n");
        content.append("=================\n");
        content.append("Node: ").append(task.nodeId().value()).append("\n");
        content.append("Status: ").append(result.status()).append("\n");
        content.append("Attempt: ").append(task.attempt()).append("\n");
        
        if (result.status() == NodeExecutionStatus.COMPLETED) {
            content.append("\nOutput:\n");
            result.output().forEach((key, value) -> {
                content.append("- ").append(key).append(": ").append(value).append("\n");
            });
        } else if (result.status() == NodeExecutionStatus.FAILED) {
            content.append("\nError:\n");
            if (result.error() != null) {
                content.append("Code: ").append(result.error().code()).append("\n");
                content.append("Message: ").append(result.error().message()).append("\n");
            }
        }
        
        return content.toString();
    }
    
    /**
     * Calculate importance score for memory
     * Higher importance = longer retention and higher retrieval priority
     */
    private double calculateImportance(NodeExecutionTask task, NodeExecutionResult result) {
        double importance = 0.5; // Base importance
        
        // Increase importance for failures (learn from mistakes)
        if (result.status() == NodeExecutionStatus.FAILED) {
            importance += 0.3;
        }
        
        // Increase importance for retries (indicates difficulty)
        if (task.attempt() > 1) {
            importance += 0.1 * Math.min(task.attempt(), 3);
        }
        
        // Increase importance for first execution
        if (task.attempt() == 1) {
            importance += 0.1;
        }
        
        return Math.min(1.0, importance);
    }
}

/**
 * ============================================================================
 * SEMANTIC MEMORY CONSOLIDATION EXECUTOR
 * ============================================================================
 * 
 * Specialized executor that consolidates episodic memories into semantic knowledge.
 * Runs periodically to:
 * - Identify patterns across episodic memories
 * - Extract semantic knowledge
 * - Reduce memory footprint
 * - Improve retrieval quality
 */

@Executor(
    executorType = "memory-consolidation",
    communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC,
    maxConcurrentTasks = 1
)
@ApplicationScoped
public class MemoryConsolidationExecutor extends AbstractWorkflowExecutor {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryConsolidationExecutor.class);
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    TextChunker textChunker;
    
    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Starting memory consolidation task");
        
        String namespace = (String) task.context().getOrDefault("namespace", "default");
        
        return consolidateMemories(namespace)
            .map(stats -> {
                Map<String, Object> output = Map.of(
                    "consolidated", stats.consolidated,
                    "patternsFound", stats.patterns,
                    "semanticMemoriesCreated", stats.semanticCreated,
                    "episodicMemoriesRetained", stats.episodicRetained
                );
                
                return NodeExecutionResult.success(
                    task.runId(),
                    task.nodeId(),
                    task.attempt(),
                    output,
                    task.token()
                );
            });
    }
    
    /**
     * Consolidate episodic memories into semantic knowledge
     */
    private Uni<ConsolidationStats> consolidateMemories(String namespace) {
        LOG.info("Consolidating memories in namespace: {}", namespace);
        
        // Get statistics
        return memoryStore.getStatistics(namespace)
            .flatMap(stats -> {
                LOG.info("Found {} episodic memories to consolidate", stats.getEpisodicCount());
                
                // For now, return basic stats
                // In real implementation, would:
                // 1. Cluster similar episodic memories
                // 2. Extract common patterns
                // 3. Create semantic memories
                // 4. Archive or delete consolidated episodic memories
                
                return Uni.createFrom().item(new ConsolidationStats(
                    stats.getEpisodicCount(),
                    0, // patterns found
                    0, // semantic created
                    stats.getEpisodicCount() // all retained for now
                ));
            });
    }
    
    /**
     * Consolidation statistics
     */
    private record ConsolidationStats(
        long consolidated,
        int patterns,
        int semanticCreated,
        long episodicRetained
    ) {}
}

/**
 * ============================================================================
 * MEMORY CLEANUP EXECUTOR
 * ============================================================================
 * 
 * Specialized executor for memory hygiene:
 * - Remove expired memories
 * - Archive old low-importance memories
 * - Compress infrequently accessed memories
 * - Maintain memory store performance
 */

@Executor(
    executorType = "memory-cleanup",
    communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC,
    maxConcurrentTasks = 1
)
@ApplicationScoped
public class MemoryCleanupExecutor extends AbstractWorkflowExecutor {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryCleanupExecutor.class);
    
    @Inject
    VectorMemoryStore memoryStore;
    
    // Cleanup thresholds
    private static final Duration MAX_WORKING_MEMORY_AGE = Duration.ofHours(24);
    private static final Duration MAX_EPISODIC_LOW_IMPORTANCE_AGE = Duration.ofDays(30);
    private static final double MIN_IMPORTANCE_THRESHOLD = 0.2;
    
    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        LOG.info("Starting memory cleanup task");
        
        String namespace = (String) task.context().getOrDefault("namespace", "default");
        
        return cleanupMemories(namespace)
            .map(stats -> {
                Map<String, Object> output = Map.of(
                    "expiredRemoved", stats.expiredRemoved,
                    "lowImportanceArchived", stats.lowImportanceArchived,
                    "workingMemoryCleared", stats.workingMemoryCleared,
                    "totalCleaned", stats.totalCleaned()
                );
                
                return NodeExecutionResult.success(
                    task.runId(),
                    task.nodeId(),
                    task.attempt(),
                    output,
                    task.token()
                );
            });
    }
    
    /**
     * Cleanup memories based on various criteria
     */
    private Uni<CleanupStats> cleanupMemories(String namespace) {
        LOG.info("Cleaning up memories in namespace: {}", namespace);
        
        return memoryStore.getStatistics(namespace)
            .flatMap(stats -> {
                // In a real implementation, would:
                // 1. Query expired memories and delete
                // 2. Query low-importance old memories and archive
                // 3. Clear working memory older than threshold
                // 4. Optimize storage
                
                LOG.info("Cleanup complete for namespace: {}", namespace);
                
                return Uni.createFrom().item(new CleanupStats(0, 0, 0));
            });
    }
    
    /**
     * Cleanup statistics
     */
    private record CleanupStats(
        int expiredRemoved,
        int lowImportanceArchived,
        int workingMemoryCleared
    ) {
        int totalCleaned() {
            return expiredRemoved + lowImportanceArchived + workingMemoryCleared;
        }
    }
}

/**
 * ============================================================================
 * MEMORY QUERY EXECUTOR
 * ============================================================================
 * 
 * Specialized executor for querying and retrieving memories.
 * Provides advanced query capabilities:
 * - Semantic search
 * - Temporal queries
 * - Metadata filtering
 * - Cross-namespace search
 */

@Executor(
    executorType = "memory-query",
    communicationType = tech.kayys.gamelan.core.scheduler.CommunicationType.GRPC,
    maxConcurrentTasks = 20
)
@ApplicationScoped
public class MemoryQueryExecutor extends AbstractWorkflowExecutor {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryQueryExecutor.class);
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    ContextEngineeringService contextService;
    
    @Override
    public Uni<NodeExecutionResult> execute(NodeExecutionTask task) {
        Map<String, Object> context = task.context();
        
        String query = (String) context.get("query");
        String namespace = (String) context.getOrDefault("namespace", "default");
        int limit = (int) context.getOrDefault("limit", 10);
        
        LOG.info("Executing memory query: '{}' in namespace: {}", query, namespace);
        
        ContextConfig config = ContextConfig.builder()
            .maxMemories(limit)
            .includeMetadata(true)
            .build();
        
        return contextService.buildContext(query, namespace, config)
            .map(engineeredContext -> {
                List<Map<String, Object>> results = new ArrayList<>();
                
                for (ContextSection section : engineeredContext.getSections()) {
                    if (section.getType().startsWith("memory_")) {
                        results.add(Map.of(
                            "content", section.getContent(),
                            "tokens", section.getTokenCount(),
                            "relevance", section.getRelevanceScore()
                        ));
                    }
                }
                
                Map<String, Object> output = Map.of(
                    "query", query,
                    "resultsCount", results.size(),
                    "results", results,
                    "contextTokens", engineeredContext.getTotalTokens()
                );
                
                return NodeExecutionResult.success(
                    task.runId(),
                    task.nodeId(),
                    task.attempt(),
                    output,
                    task.token()
                );
            });
    }
}

package tech.kayys.gamelan.executor.memory;

import io.quarkus.hibernate.reactive.panache.Panache;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.pgclient.PgPool;
import io.vertx.mutiny.sqlclient.Row;
import io.vertx.mutiny.sqlclient.RowSet;
import io.vertx.mutiny.sqlclient.Tuple;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.persistence.*;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.hibernate.annotations.Type;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * ============================================================================
 * POSTGRESQL PGVECTOR MEMORY STORE
 * ============================================================================
 * 
 * Production-grade vector memory store using PostgreSQL with pgvector extension.
 * 
 * Features:
 * - Efficient vector similarity search using HNSW or IVFFlat indexes
 * - Hybrid search combining vector and full-text search
 * - Metadata filtering with JSONB
 * - Horizontal scaling with partitioning
 * - Multi-tenancy support
 * 
 * Prerequisites:
 * - PostgreSQL 12+ with pgvector extension
 * - CREATE EXTENSION vector;
 */

@ApplicationScoped
public class PostgresVectorStore implements VectorMemoryStore {
    
    private static final Logger LOG = LoggerFactory.getLogger(PostgresVectorStore.class);
    
    @Inject
    PgPool pgPool;
    
    @ConfigProperty(name = "gamelan.memory.vector.dimension", defaultValue = "1536")
    int vectorDimension;
    
    @ConfigProperty(name = "gamelan.memory.index.type", defaultValue = "hnsw")
    String indexType; // hnsw or ivfflat
    
    /**
     * Initialize database schema
     */
    public Uni<Void> initialize() {
        LOG.info("Initializing PostgreSQL vector store");
        
        String createTableSql = """
            CREATE TABLE IF NOT EXISTS gamelan_memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                embedding vector(%d),
                type VARCHAR(50) NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ,
                importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                tenant_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON gamelan_memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_tenant ON gamelan_memories(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON gamelan_memories(type);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON gamelan_memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON gamelan_memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON gamelan_memories USING GIN(content_tsvector);
            CREATE INDEX IF NOT EXISTS idx_memories_metadata ON gamelan_memories USING GIN(metadata);
            
            -- Vector similarity index (HNSW for better performance)
            CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
                ON gamelan_memories USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """.formatted(vectorDimension);
        
        return pgPool.query(createTableSql)
            .execute()
            .replaceWithVoid()
            .invoke(() -> LOG.info("Vector store initialized"));
    }
    
    @Override
    public Uni<String> store(Memory memory) {
        LOG.debug("Storing memory: {}", memory.getId());
        
        String sql = """
            INSERT INTO gamelan_memories (
                id, namespace, content, embedding, type, metadata,
                timestamp, expires_at, importance, tenant_id
            ) VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                importance = EXCLUDED.importance,
                updated_at = NOW()
            RETURNING id
            """;
        
        Tuple params = Tuple.of(
            UUID.fromString(memory.getId()),
            memory.getNamespace(),
            memory.getContent(),
            vectorToString(memory.getEmbedding()),
            memory.getType().name(),
            toJsonb(memory.getMetadata()),
            memory.getTimestamp(),
            memory.getExpiresAt(),
            memory.getImportance(),
            extractRequestId(memory.getNamespace())
        );
        
        return pgPool.preparedQuery(sql)
            .execute(params)
            .map(rowSet -> {
                Row row = rowSet.iterator().next();
                return row.getUUID("id").toString();
            });
    }
    
    @Override
    public Uni<List<String>> storeBatch(List<Memory> memories) {
        LOG.debug("Storing batch of {} memories", memories.size());
        
        return Panache.withTransaction(() -> {
            List<Uni<String>> stores = memories.stream()
                .map(this::store)
                .toList();
            
            return Uni.join().all(stores).andFailFast();
        });
    }
    
    @Override
    public Uni<List<ScoredMemory>> search(
            float[] queryEmbedding,
            int limit,
            double minSimilarity,
            Map<String, Object> filters) {
        
        LOG.debug("Vector search with limit: {}, minSimilarity: {}", limit, minSimilarity);
        
        // Build dynamic query based on filters
        StringBuilder sql = new StringBuilder("""
            SELECT 
                id, namespace, content, embedding::text, type, 
                metadata, timestamp, expires_at, importance,
                1 - (embedding <=> $1::vector) as similarity
            FROM gamelan_memories
            WHERE 1=1
            """);
        
        List<Object> params = new ArrayList<>();
        params.add(vectorToString(queryEmbedding));
        int paramIndex = 2;
        
        // Add filters
        if (filters != null) {
            if (filters.containsKey("namespace")) {
                sql.append(" AND namespace = $").append(paramIndex++);
                params.add(filters.get("namespace"));
            }
            
            if (filters.containsKey("types")) {
                sql.append(" AND type = ANY($").append(paramIndex++).append(")");
                List<String> types = ((List<?>) filters.get("types")).stream()
                    .map(Object::toString)
                    .toList();
                params.add(types.toArray(new String[0]));
            }
            
            if (filters.containsKey("minImportance")) {
                sql.append(" AND importance >= $").append(paramIndex++);
                params.add(filters.get("minImportance"));
            }
        }
        
        // Exclude expired
        sql.append(" AND (expires_at IS NULL OR expires_at > NOW())");
        
        // Similarity threshold
        sql.append(" AND 1 - (embedding <=> $1::vector) >= $").append(paramIndex++);
        params.add(minSimilarity);
        
        // Order and limit
        sql.append(" ORDER BY embedding <=> $1::vector ASC LIMIT $").append(paramIndex);
        params.add(limit);
        
        return pgPool.preparedQuery(sql.toString())
            .execute(Tuple.wrap(params))
            .map(rowSet -> {
                List<ScoredMemory> results = new ArrayList<>();
                
                for (Row row : rowSet) {
                    Memory memory = rowToMemory(row);
                    double similarity = row.getDouble("similarity");
                    results.add(new ScoredMemory(memory, similarity));
                }
                
                LOG.debug("Found {} memories", results.size());
                return results;
            });
    }
    
    @Override
    public Uni<List<ScoredMemory>> hybridSearch(
            float[] queryEmbedding,
            List<String> keywords,
            int limit,
            double semanticWeight) {
        
        LOG.debug("Hybrid search with {} keywords, semantic weight: {}", 
            keywords.size(), semanticWeight);
        
        String keywordQuery = String.join(" | ", keywords);
        double keywordWeight = 1.0 - semanticWeight;
        
        String sql = """
            WITH semantic_results AS (
                SELECT 
                    id, namespace, content, embedding::text, type,
                    metadata, timestamp, expires_at, importance,
                    1 - (embedding <=> $1::vector) as semantic_score
                FROM gamelan_memories
                WHERE (expires_at IS NULL OR expires_at > NOW())
                    AND namespace = $2
            ),
            keyword_results AS (
                SELECT 
                    id,
                    ts_rank(content_tsvector, to_tsquery('english', $3)) as keyword_score
                FROM gamelan_memories
                WHERE content_tsvector @@ to_tsquery('english', $3)
                    AND (expires_at IS NULL OR expires_at > NOW())
                    AND namespace = $2
            )
            SELECT 
                s.id, s.namespace, s.content, s.embedding, s.type,
                s.metadata, s.timestamp, s.expires_at, s.importance,
                (s.semantic_score * $4 + COALESCE(k.keyword_score, 0) * $5) as combined_score,
                s.semantic_score,
                COALESCE(k.keyword_score, 0) as keyword_score
            FROM semantic_results s
            LEFT JOIN keyword_results k ON s.id = k.id
            ORDER BY combined_score DESC
            LIMIT $6
            """;
        
        String namespace = "default"; // From filters if available
        
        Tuple params = Tuple.of(
            vectorToString(queryEmbedding),
            namespace,
            keywordQuery,
            semanticWeight,
            keywordWeight,
            limit
        );
        
        return pgPool.preparedQuery(sql)
            .execute(params)
            .map(rowSet -> {
                List<ScoredMemory> results = new ArrayList<>();
                
                for (Row row : rowSet) {
                    Memory memory = rowToMemory(row);
                    double combinedScore = row.getDouble("combined_score");
                    
                    Map<String, Object> scoreBreakdown = Map.of(
                        "total", combinedScore,
                        "semantic", row.getDouble("semantic_score"),
                        "keyword", row.getDouble("keyword_score")
                    );
                    
                    results.add(new ScoredMemory(memory, combinedScore, scoreBreakdown));
                }
                
                return results;
            });
    }
    
    @Override
    public Uni<Memory> retrieve(String memoryId) {
        LOG.debug("Retrieving memory: {}", memoryId);
        
        String sql = """
            SELECT id, namespace, content, embedding::text, type,
                   metadata, timestamp, expires_at, importance
            FROM gamelan_memories
            WHERE id = $1
                AND (expires_at IS NULL OR expires_at > NOW())
            """;
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of(UUID.fromString(memoryId)))
            .map(rowSet -> {
                if (!rowSet.iterator().hasNext()) {
                    return null;
                }
                return rowToMemory(rowSet.iterator().next());
            });
    }
    
    @Override
    public Uni<List<Memory>> retrieveBatch(List<String> memoryIds) {
        LOG.debug("Retrieving batch of {} memories", memoryIds.size());
        
        String sql = """
            SELECT id, namespace, content, embedding::text, type,
                   metadata, timestamp, expires_at, importance
            FROM gamelan_memories
            WHERE id = ANY($1)
                AND (expires_at IS NULL OR expires_at > NOW())
            """;
        
        UUID[] uuids = memoryIds.stream()
            .map(UUID::fromString)
            .toArray(UUID[]::new);
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of((Object) uuids))
            .map(rowSet -> {
                List<Memory> memories = new ArrayList<>();
                for (Row row : rowSet) {
                    memories.add(rowToMemory(row));
                }
                return memories;
            });
    }
    
    @Override
    public Uni<Memory> updateMetadata(String memoryId, Map<String, Object> metadata) {
        LOG.debug("Updating metadata for memory: {}", memoryId);
        
        String sql = """
            UPDATE gamelan_memories
            SET metadata = metadata || $1::jsonb,
                updated_at = NOW()
            WHERE id = $2
            RETURNING id, namespace, content, embedding::text, type,
                      metadata, timestamp, expires_at, importance
            """;
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of(toJsonb(metadata), UUID.fromString(memoryId)))
            .map(rowSet -> {
                if (!rowSet.iterator().hasNext()) {
                    return null;
                }
                return rowToMemory(rowSet.iterator().next());
            });
    }
    
    @Override
    public Uni<Boolean> delete(String memoryId) {
        LOG.debug("Deleting memory: {}", memoryId);
        
        String sql = "DELETE FROM gamelan_memories WHERE id = $1";
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of(UUID.fromString(memoryId)))
            .map(rowSet -> rowSet.rowCount() > 0);
    }
    
    @Override
    public Uni<Long> deleteNamespace(String namespace) {
        LOG.info("Deleting all memories in namespace: {}", namespace);
        
        String sql = "DELETE FROM gamelan_memories WHERE namespace = $1";
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of(namespace))
            .map(rowSet -> (long) rowSet.rowCount());
    }
    
    @Override
    public Uni<MemoryStatistics> getStatistics(String namespace) {
        LOG.debug("Getting statistics for namespace: {}", namespace);
        
        String sql = """
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE type = 'EPISODIC') as episodic,
                COUNT(*) FILTER (WHERE type = 'SEMANTIC') as semantic,
                COUNT(*) FILTER (WHERE type = 'PROCEDURAL') as procedural,
                COUNT(*) FILTER (WHERE type = 'WORKING') as working,
                AVG(importance) as avg_importance,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest
            FROM gamelan_memories
            WHERE namespace = $1
                AND (expires_at IS NULL OR expires_at > NOW())
            """;
        
        return pgPool.preparedQuery(sql)
            .execute(Tuple.of(namespace))
            .map(rowSet -> {
                if (!rowSet.iterator().hasNext()) {
                    return new MemoryStatistics(namespace, 0, 0, 0, 0, 0, 0.0, null, null);
                }
                
                Row row = rowSet.iterator().next();
                
                return new MemoryStatistics(
                    namespace,
                    row.getLong("total"),
                    row.getLong("episodic"),
                    row.getLong("semantic"),
                    row.getLong("procedural"),
                    row.getLong("working"),
                    row.getDouble("avg_importance"),
                    row.getLocalDateTime("oldest") != null ? 
                        Instant.from(row.getLocalDateTime("oldest")) : null,
                    row.getLocalDateTime("newest") != null ?
                        Instant.from(row.getLocalDateTime("newest")) : null
                );
            });
    }
    
    // ==================== UTILITY METHODS ====================
    
    /**
     * Convert row to Memory object
     */
    private Memory rowToMemory(Row row) {
        return Memory.builder()
            .id(row.getUUID("id").toString())
            .namespace(row.getString("namespace"))
            .content(row.getString("content"))
            .embedding(stringToVector(row.getString("embedding")))
            .type(MemoryType.valueOf(row.getString("type")))
            .metadata(fromJsonb(row.getString("metadata")))
            .timestamp(Instant.from(row.getLocalDateTime("timestamp")))
            .expiresAt(row.getLocalDateTime("expires_at") != null ?
                Instant.from(row.getLocalDateTime("expires_at")) : null)
            .importance(row.getDouble("importance"))
            .build();
    }
    
    /**
     * Convert float array to PostgreSQL vector format
     */
    private String vectorToString(float[] vector) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(vector[i]);
        }
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * Convert PostgreSQL vector string to float array
     */
    private float[] stringToVector(String vectorStr) {
        // Remove brackets and split
        String cleaned = vectorStr.substring(1, vectorStr.length() - 1);
        String[] parts = cleaned.split(",");
        
        float[] vector = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            vector[i] = Float.parseFloat(parts[i].trim());
        }
        
        return vector;
    }
    
    /**
     * Convert map to JSONB string
     */
    private String toJsonb(Map<String, Object> map) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                .writeValueAsString(map);
        } catch (Exception e) {
            LOG.error("Failed to convert map to JSONB", e);
            return "{}";
        }
    }
    
    /**
     * Convert JSONB string to map
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> fromJsonb(String jsonb) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                .readValue(jsonb, Map.class);
        } catch (Exception e) {
            LOG.error("Failed to parse JSONB", e);
            return new HashMap<>();
        }
    }
    
    /**
     * Extract tenant ID from namespace
     */
    private String extractRequestId(String namespace) {
        // Namespace format: tenant:workflow:node
        String[] parts = namespace.split(":");
        return parts.length > 0 ? parts[0] : "default";
    }
}

/**
 * ============================================================================
 * VECTOR STORE FACTORY
 * ============================================================================
 * 
 * Factory for creating appropriate vector store based on configuration
 */

@ApplicationScoped
public class VectorStoreFactory {
    
    private static final Logger LOG = LoggerFactory.getLogger(VectorStoreFactory.class);
    
    @ConfigProperty(name = "gamelan.memory.store.type", defaultValue = "inmemory")
    String storeType;
    
    @Inject
    InMemoryVectorStore inMemoryStore;
    
    @Inject
    PostgresVectorStore postgresStore;
    
    /**
     * Get configured vector store
     */
    public VectorMemoryStore getVectorStore() {
        LOG.info("Using vector store type: {}", storeType);
        
        VectorMemoryStore store = switch (storeType.toLowerCase()) {
            case "postgres", "postgresql", "pgvector" -> postgresStore;
            case "inmemory", "memory" -> inMemoryStore;
            default -> {
                LOG.warn("Unknown store type: {}, falling back to in-memory", storeType);
                yield inMemoryStore;
            }
        };
        
        // Initialize if needed
        if (store instanceof PostgresVectorStore pgStore) {
            pgStore.initialize()
                .subscribe().with(
                    v -> LOG.info("Vector store initialized"),
                    error -> LOG.error("Failed to initialize vector store", error)
                );
        }
        
        return store;
    }
}


------
# ============================================================================
# GAMELAN MEMORY EXECUTOR CONFIGURATION
# ============================================================================

# Application Info
quarkus.application.name=gamelan-memory-executor
quarkus.application.version=1.0.0

# HTTP Configuration
quarkus.http.port=8081
quarkus.http.host=0.0.0.0
quarkus.http.cors=true

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# PostgreSQL with pgvector
quarkus.datasource.db-kind=postgresql
quarkus.datasource.username=${DB_USER:gamelan}
quarkus.datasource.password=${DB_PASSWORD:gamelan}
quarkus.datasource.reactive.url=postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME:gamelan_memory}

# Connection Pool
quarkus.datasource.reactive.max-size=20
quarkus.datasource.reactive.idle-timeout=PT10M

# Hibernate
quarkus.hibernate-orm.database.generation=update
quarkus.hibernate-orm.log.sql=false
quarkus.hibernate-orm.sql-load-script=no-file

# ============================================================================
# MEMORY STORE CONFIGURATION
# ============================================================================

# Store Type: inmemory, postgres, pinecone, weaviate
gamelan.memory.store.type=${MEMORY_STORE_TYPE:inmemory}

# Vector Configuration
gamelan.memory.vector.dimension=${VECTOR_DIMENSION:1536}
gamelan.memory.index.type=${VECTOR_INDEX_TYPE:hnsw}

# ============================================================================
# EMBEDDING SERVICE CONFIGURATION
# ============================================================================

# Provider: openai, cohere, local
gamelan.embedding.provider=${EMBEDDING_PROVIDER:local}

# OpenAI Configuration
gamelan.embedding.openai.api-key=${OPENAI_API_KEY:}
gamelan.embedding.openai.model=${OPENAI_EMBEDDING_MODEL:text-embedding-3-small}
gamelan.embedding.openai.endpoint=https://api.openai.com/v1/embeddings

# Cache Configuration
gamelan.embedding.cache.enabled=${EMBEDDING_CACHE_ENABLED:true}
gamelan.embedding.cache.max-size=${EMBEDDING_CACHE_SIZE:10000}

# Text Chunking
gamelan.embedding.chunk.size=${CHUNK_SIZE:512}
gamelan.embedding.chunk.overlap=${CHUNK_OVERLAP:50}

# ============================================================================
# CONTEXT ENGINEERING CONFIGURATION
# ============================================================================

# Context Window
gamelan.context.max-tokens=${CONTEXT_MAX_TOKENS:8000}

# Scoring Weights
gamelan.context.decay-rate=${CONTEXT_DECAY_RATE:0.001}
gamelan.context.recency-weight=${CONTEXT_RECENCY_WEIGHT:0.3}
gamelan.context.importance-weight=${CONTEXT_IMPORTANCE_WEIGHT:0.4}
gamelan.context.similarity-weight=${CONTEXT_SIMILARITY_WEIGHT:0.3}

# ============================================================================
# EXECUTOR CONFIGURATION
# ============================================================================

# Executor Registration
gamelan.executor.id=${EXECUTOR_ID:memory-executor-1}
gamelan.executor.transport=${EXECUTOR_TRANSPORT:GRPC}

# Engine Connection
gamelan.engine.grpc.host=${ENGINE_GRPC_HOST:localhost}
gamelan.engine.grpc.port=${ENGINE_GRPC_PORT:9090}
gamelan.engine.rest.endpoint=${ENGINE_REST_ENDPOINT:http://localhost:8080}

# Kafka Configuration (if using Kafka transport)
kafka.bootstrap.servers=${KAFKA_BOOTSTRAP_SERVERS:localhost:9092}
mp.messaging.incoming.workflow-tasks.connector=smallrye-kafka
mp.messaging.incoming.workflow-tasks.topic=gamelan.tasks
mp.messaging.incoming.workflow-tasks.value.deserializer=org.apache.kafka.common.serialization.StringDeserializer
mp.messaging.incoming.workflow-tasks.group.id=gamelan-memory-executors

mp.messaging.outgoing.workflow-results.connector=smallrye-kafka
mp.messaging.outgoing.workflow-results.topic=gamelan.results
mp.messaging.outgoing.workflow-results.value.serializer=org.apache.kafka.common.serialization.StringSerializer

# ============================================================================
# MULTI-TENANCY CONFIGURATION
# ============================================================================

# Tenant Isolation
gamelan.multitenancy.enabled=${MULTITENANCY_ENABLED:true}
gamelan.multitenancy.default-tenant=${DEFAULT_TENANT:default}

# ============================================================================
# OBSERVABILITY
# ============================================================================

# Logging
quarkus.log.level=INFO
quarkus.log.category."tech.kayys.gamelan".level=DEBUG
quarkus.log.console.format=%d{yyyy-MM-dd HH:mm:ss,SSS} %-5p [%c{2.}] (%t) %s%e%n

# Health Checks
quarkus.smallrye-health.root-path=/health
quarkus.smallrye-health.liveness-path=/health/live
quarkus.smallrye-health.readiness-path=/health/ready

# Metrics
quarkus.micrometer.enabled=true
quarkus.micrometer.registry-enabled-default=true
quarkus.micrometer.export.prometheus.enabled=true
quarkus.micrometer.export.prometheus.path=/metrics

# ============================================================================
# REST CLIENT CONFIGURATION
# ============================================================================

# OpenAI REST Client
quarkus.rest-client.openai.url=https://api.openai.com
quarkus.rest-client.openai.scope=jakarta.inject.Singleton
quarkus.rest-client.openai.connect-timeout=30000
quarkus.rest-client.openai.read-timeout=60000

# ============================================================================
# DEVELOPMENT CONFIGURATION
# ============================================================================

%dev.quarkus.log.console.level=DEBUG
%dev.gamelan.memory.store.type=inmemory
%dev.gamelan.embedding.provider=local
%dev.quarkus.datasource.reactive.url=postgresql://localhost:5432/gamelan_memory_dev

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

%test.quarkus.datasource.db-kind=h2
%test.quarkus.datasource.reactive.url=jdbc:h2:mem:test;DB_CLOSE_DELAY=-1
%test.gamelan.memory.store.type=inmemory
%test.gamelan.embedding.provider=local

# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================

%prod.quarkus.log.level=INFO
%prod.quarkus.log.category."tech.kayys.gamelan".level=INFO
%prod.gamelan.memory.store.type=postgres
%prod.gamelan.embedding.provider=openai


-------

package tech.kayys.gamelan.executor.memory.examples;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gamelan.executor.memory.*;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * ============================================================================
 * MEMORY EXECUTOR USAGE EXAMPLES
 * ============================================================================
 * 
 * Comprehensive examples demonstrating the memory executor capabilities.
 */

@ApplicationScoped
public class MemoryExecutorExamples {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorExamples.class);
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    ContextEngineeringService contextService;
    
    /**
     * Example 1: Store and Retrieve Memories
     */
    public Uni<Void> example1_BasicMemoryOperations() {
        LOG.info("=== Example 1: Basic Memory Operations ===");
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // Create a memory about a customer interaction
        String content = """
            Customer John Doe contacted support regarding order #12345.
            Issue: Product arrived damaged.
            Resolution: Full refund processed, replacement shipped.
            Customer satisfaction: High
            """;
        
        return embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("customer-support")
                    .content(content)
                    .embedding(embedding)
                    .type(MemoryType.EPISODIC)
                    .importance(0.8)
                    .addMetadata("customerId", "CUST-001")
                    .addMetadata("orderId", "ORD-12345")
                    .addMetadata("category", "refund")
                    .addMetadata("sentiment", "positive")
                    .build();
                
                return memoryStore.store(memory);
            })
            .flatMap(memoryId -> {
                LOG.info("Stored memory: {}", memoryId);
                
                // Retrieve the memory
                return memoryStore.retrieve(memoryId);
            })
            .invoke(retrieved -> {
                LOG.info("Retrieved memory: {}", retrieved.getContent());
                LOG.info("Importance: {}", retrieved.getImportance());
                LOG.info("Metadata: {}", retrieved.getMetadata());
            })
            .replaceWithVoid();
    }
    
    /**
     * Example 2: Semantic Search
     */
    public Uni<Void> example2_SemanticSearch() {
        LOG.info("=== Example 2: Semantic Search ===");
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // First, populate with some memories
        List<String> customerIssues = List.of(
            "Product arrived damaged, customer wants refund",
            "Delivery delayed by 2 days, customer upset",
            "Wrong item shipped, customer needs replacement",
            "Product quality excellent, customer very happy",
            "Customer cannot login to account, needs password reset"
        );
        
        return Uni.join().all(
            customerIssues.stream()
                .map(issue -> storeCustomerIssue(issue))
                .toList()
        ).andFailFast()
        .flatMap(stored -> {
            LOG.info("Stored {} customer issues", stored.size());
            
            // Now search for similar issues
            String query = "Customer received broken product";
            
            return embeddingService.embed(query)
                .flatMap(queryEmbedding -> 
                    memoryStore.search(
                        queryEmbedding,
                        3, // Top 3 results
                        0.5, // Min similarity
                        Map.of("namespace", "customer-support")
                    )
                );
        })
        .invoke(results -> {
            LOG.info("Found {} similar issues:", results.size());
            results.forEach(scored -> {
                LOG.info("  - Score: {:.3f} | {}", 
                    scored.getScore(), 
                    scored.getMemory().getContent().substring(0, 50) + "...");
            });
        })
        .replaceWithVoid();
    }
    
    /**
     * Example 3: Context Engineering
     */
    public Uni<Void> example3_ContextEngineering() {
        LOG.info("=== Example 3: Context Engineering ===");
        
        // Populate knowledge base
        List<String> knowledgeArticles = List.of(
            "To process a refund: 1) Verify order number, 2) Check refund policy, 3) Process via payment system",
            "Damaged products qualify for full refund within 30 days of delivery",
            "Replacement shipping is free for damaged items",
            "Customer satisfaction is our top priority",
            "All refunds are processed within 3-5 business days"
        );
        
        return Uni.join().all(
            knowledgeArticles.stream()
                .map(article -> storeKnowledgeArticle(article))
                .toList()
        ).andFailFast()
        .flatMap(stored -> {
            LOG.info("Stored {} knowledge articles", stored.size());
            
            // Build context for a customer support query
            String query = "How do I handle a damaged product complaint?";
            
            ContextConfig config = ContextConfig.builder()
                .maxMemories(3)
                .systemPrompt("You are a helpful customer support assistant.")
                .taskInstructions("Provide clear step-by-step guidance.")
                .memoryTypes(List.of(MemoryType.SEMANTIC, MemoryType.PROCEDURAL))
                .includeMetadata(false)
                .build();
            
            return contextService.buildContext(query, "customer-support", config);
        })
        .invoke(context -> {
            LOG.info("Context built successfully:");
            LOG.info("  Total tokens: {}", context.getTotalTokens());
            LOG.info("  Utilization: {:.1f}%", context.getUtilization() * 100);
            LOG.info("  Sections: {}", context.getSections().size());
            
            LOG.info("\n=== Generated Prompt ===");
            LOG.info(context.toPrompt());
        })
        .replaceWithVoid();
    }
    
    /**
     * Example 4: Hybrid Search (Semantic + Keyword)
     */
    public Uni<Void> example4_HybridSearch() {
        LOG.info("=== Example 4: Hybrid Search ===");
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        String query = "refund policy for damaged items";
        List<String> keywords = List.of("refund", "damaged", "policy");
        
        return embeddingService.embed(query)
            .flatMap(queryEmbedding ->
                memoryStore.hybridSearch(
                    queryEmbedding,
                    keywords,
                    5,
                    0.7 // 70% semantic, 30% keyword
                )
            )
            .invoke(results -> {
                LOG.info("Hybrid search results:");
                results.forEach(scored -> {
                    LOG.info("  Combined Score: {:.3f}", scored.getScore());
                    LOG.info("    Breakdown: {}", scored.getScoreBreakdown());
                    LOG.info("    Content: {}...\n", 
                        scored.getMemory().getContent().substring(0, 60));
                });
            })
            .replaceWithVoid();
    }
    
    /**
     * Example 5: Memory with Temporal Decay
     */
    public Uni<Void> example5_TemporalDecay() {
        LOG.info("=== Example 5: Temporal Decay ===");
        
        // Create memories at different times
        Instant now = Instant.now();
        
        List<Uni<String>> memoryOps = new ArrayList<>();
        
        // Recent memory (1 hour ago)
        memoryOps.add(createTimedMemory(
            "Recent customer interaction - very helpful",
            now.minus(Duration.ofHours(1)),
            0.8
        ));
        
        // Older memory (1 day ago)
        memoryOps.add(createTimedMemory(
            "Yesterday's interaction - also helpful",
            now.minus(Duration.ofDays(1)),
            0.8
        ));
        
        // Very old memory (30 days ago)
        memoryOps.add(createTimedMemory(
            "Old interaction from last month",
            now.minus(Duration.ofDays(30)),
            0.8
        ));
        
        return Uni.join().all(memoryOps).andFailFast()
            .flatMap(memoryIds -> {
                // Retrieve all and show decayed importance
                return memoryStore.retrieveBatch(memoryIds);
            })
            .invoke(memories -> {
                LOG.info("Memory importance with temporal decay (rate=0.001):");
                
                memories.forEach(memory -> {
                    Duration age = Duration.between(memory.getTimestamp(), Instant.now());
                    double decayed = memory.getDecayedImportance(0.001);
                    
                    LOG.info("  Age: {} hours | Original: {:.3f} | Decayed: {:.3f}",
                        age.toHours(),
                        memory.getImportance(),
                        decayed
                    );
                });
            })
            .replaceWithVoid();
    }
    
    /**
     * Example 6: Batch Operations
     */
    public Uni<Void> example6_BatchOperations() {
        LOG.info("=== Example 6: Batch Operations ===");
        
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        List<String> contents = List.of(
            "Customer inquiry about shipping times",
            "Product return request received",
            "Payment processing issue reported",
            "Account registration problem",
            "Newsletter subscription confirmed"
        );
        
        // Batch embed
        return embeddingService.embedBatch(contents)
            .flatMap(embeddings -> {
                LOG.info("Generated {} embeddings in batch", embeddings.size());
                
                // Create memories
                List<Memory> memories = new ArrayList<>();
                for (int i = 0; i < contents.size(); i++) {
                    memories.add(Memory.builder()
                        .namespace("customer-support")
                        .content(contents.get(i))
                        .embedding(embeddings.get(i))
                        .type(MemoryType.EPISODIC)
                        .importance(0.5 + (i * 0.1))
                        .build()
                    );
                }
                
                // Batch store
                return memoryStore.storeBatch(memories);
            })
            .invoke(memoryIds -> {
                LOG.info("Stored {} memories in batch", memoryIds.size());
                memoryIds.forEach(id -> LOG.info("  - {}", id));
            })
            .replaceWithVoid();
    }
    
    /**
     * Example 7: Memory Statistics
     */
    public Uni<Void> example7_MemoryStatistics() {
        LOG.info("=== Example 7: Memory Statistics ===");
        
        return memoryStore.getStatistics("customer-support")
            .invoke(stats -> {
                LOG.info("Memory Statistics for 'customer-support':");
                LOG.info("  Total Memories: {}", stats.getTotalMemories());
                LOG.info("  Episodic: {}", stats.getEpisodicCount());
                LOG.info("  Semantic: {}", stats.getSemanticCount());
                LOG.info("  Procedural: {}", stats.getProceduralCount());
                LOG.info("  Working: {}", stats.getWorkingCount());
                LOG.info("  Avg Importance: {:.3f}", stats.getAvgImportance());
                LOG.info("  Oldest Memory: {}", stats.getOldestMemory());
                LOG.info("  Newest Memory: {}", stats.getNewestMemory());
            })
            .replaceWithVoid();
    }
    
    // ==================== HELPER METHODS ====================
    
    private Uni<String> storeCustomerIssue(String content) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        return embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("customer-support")
                    .content(content)
                    .embedding(embedding)
                    .type(MemoryType.EPISODIC)
                    .importance(0.6)
                    .addMetadata("category", "issue")
                    .build();
                
                return memoryStore.store(memory);
            });
    }
    
    private Uni<String> storeKnowledgeArticle(String content) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        return embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("customer-support")
                    .content(content)
                    .embedding(embedding)
                    .type(MemoryType.SEMANTIC)
                    .importance(0.9)
                    .addMetadata("category", "knowledge")
                    .build();
                
                return memoryStore.store(memory);
            });
    }
    
    private Uni<String> createTimedMemory(String content, Instant timestamp, double importance) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        return embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("customer-support")
                    .content(content)
                    .embedding(embedding)
                    .type(MemoryType.EPISODIC)
                    .timestamp(timestamp)
                    .importance(importance)
                    .build();
                
                return memoryStore.store(memory);
            });
    }
}

/**
 * ============================================================================
 * EXAMPLE RUNNER
 * ============================================================================
 */

@ApplicationScoped
public class ExampleRunner {
    
    private static final Logger LOG = LoggerFactory.getLogger(ExampleRunner.class);
    
    @Inject
    MemoryExecutorExamples examples;
    
    /**
     * Run all examples sequentially
     */
    public Uni<Void> runAllExamples() {
        LOG.info("Starting Memory Executor Examples...\n");
        
        return examples.example1_BasicMemoryOperations()
            .chain(() -> examples.example2_SemanticSearch())
            .chain(() -> examples.example3_ContextEngineering())
            .chain(() -> examples.example4_HybridSearch())
            .chain(() -> examples.example5_TemporalDecay())
            .chain(() -> examples.example6_BatchOperations())
            .chain(() -> examples.example7_MemoryStatistics())
            .invoke(() -> LOG.info("\n=== All Examples Completed Successfully ==="));
    }
}

------
version: '3.8'

services:
  # PostgreSQL with pgvector
  postgres:
    image: pgvector/pgvector:pg16
    container_name: gamelan-postgres
    environment:
      POSTGRES_DB: gamelan_memory
      POSTGRES_USER: gamelan
      POSTGRES_PASSWORD: gamelan
      POSTGRES_INITDB_ARGS: "-E UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gamelan"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - gamelan-network

  # Gamelan Memory Executor
  memory-executor:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gamelan-memory-executor
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      # Database
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: gamelan_memory
      DB_USER: gamelan
      DB_PASSWORD: gamelan
      
      # Memory Store
      MEMORY_STORE_TYPE: postgres
      VECTOR_DIMENSION: 1536
      VECTOR_INDEX_TYPE: hnsw
      
      # Embedding Service
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-local}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      
      # Context Engineering
      CONTEXT_MAX_TOKENS: 8000
      CONTEXT_DECAY_RATE: 0.001
      
      # Executor
      EXECUTOR_ID: memory-executor-1
      EXECUTOR_TRANSPORT: GRPC
      ENGINE_GRPC_HOST: gamelan-engine
      ENGINE_GRPC_PORT: 9090
      
      # Multi-tenancy
      MULTITENANCY_ENABLED: true
      DEFAULT_TENANT: default
      
      # Java Options
      JAVA_OPTS: "-Xms512m -Xmx2048m -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
    ports:
      - "8081:8081"
      - "9091:9090"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health/live"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    networks:
      - gamelan-network
    restart: unless-stopped

  # Kafka (optional, for Kafka-based executors)
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    container_name: gamelan-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - gamelan-network
    profiles:
      - kafka

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    container_name: gamelan-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    networks:
      - gamelan-network
    profiles:
      - kafka

  # Prometheus (monitoring)
  prometheus:
    image: prom/prometheus:v2.48.1
    container_name: gamelan-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    networks:
      - gamelan-network
    profiles:
      - monitoring

  # Grafana (visualization)
  grafana:
    image: grafana/grafana:10.2.3
    container_name: gamelan-grafana
    depends_on:
      - prometheus
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - gamelan-network
    profiles:
      - monitoring

volumes:
  postgres-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  gamelan-network:
    driver: bridge

--------------------------------------


-- ============================================================================
-- GAMELAN MEMORY STORE DATABASE INITIALIZATION
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema
CREATE SCHEMA IF NOT EXISTS gamelan;

-- Set search path
SET search_path TO gamelan, public;

-- ============================================================================
-- MAIN MEMORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS gamelan_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(1536),
    type VARCHAR(50) NOT NULL CHECK (type IN ('EPISODIC', 'SEMANTIC', 'PROCEDURAL', 'WORKING')),
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Namespace index
CREATE INDEX IF NOT EXISTS idx_memories_namespace 
    ON gamelan_memories(namespace);

-- Tenant index
CREATE INDEX IF NOT EXISTS idx_memories_tenant 
    ON gamelan_memories(tenant_id);

-- Type index
CREATE INDEX IF NOT EXISTS idx_memories_type 
    ON gamelan_memories(type);

-- Timestamp index (for temporal queries)
CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
    ON gamelan_memories(timestamp DESC);

-- Importance index
CREATE INDEX IF NOT EXISTS idx_memories_importance 
    ON gamelan_memories(importance DESC);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memories_content_fts 
    ON gamelan_memories USING GIN(content_tsvector);

-- JSONB metadata index
CREATE INDEX IF NOT EXISTS idx_memories_metadata 
    ON gamelan_memories USING GIN(metadata);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_memories_tenant_namespace_timestamp
    ON gamelan_memories(tenant_id, namespace, timestamp DESC);

-- Vector similarity index (HNSW for best performance)
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
    ON gamelan_memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (faster build, slower search)
-- CREATE INDEX IF NOT EXISTS idx_memories_embedding_ivfflat 
--     ON gamelan_memories USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- ============================================================================
-- PARTITIONING (Optional, for large-scale deployments)
-- ============================================================================

-- Uncomment to enable partitioning by tenant
-- ALTER TABLE gamelan_memories PARTITION BY LIST (tenant_id);

-- Create default partition
-- CREATE TABLE gamelan_memories_default PARTITION OF gamelan_memories DEFAULT;

-- Example: Create partition for specific tenant
-- CREATE TABLE gamelan_memories_acme PARTITION OF gamelan_memories
--     FOR VALUES IN ('acme-corp');

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updated_at
CREATE TRIGGER update_gamelan_memories_updated_at
    BEFORE UPDATE ON gamelan_memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to clean expired memories
CREATE OR REPLACE FUNCTION clean_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM gamelan_memories
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get memory statistics
CREATE OR REPLACE FUNCTION get_memory_stats(p_namespace VARCHAR)
RETURNS TABLE (
    total_memories BIGINT,
    episodic_count BIGINT,
    semantic_count BIGINT,
    procedural_count BIGINT,
    working_count BIGINT,
    avg_importance DOUBLE PRECISION,
    oldest_memory TIMESTAMPTZ,
    newest_memory TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_memories,
        COUNT(*) FILTER (WHERE type = 'EPISODIC') as episodic_count,
        COUNT(*) FILTER (WHERE type = 'SEMANTIC') as semantic_count,
        COUNT(*) FILTER (WHERE type = 'PROCEDURAL') as procedural_count,
        COUNT(*) FILTER (WHERE type = 'WORKING') as working_count,
        AVG(importance) as avg_importance,
        MIN(timestamp) as oldest_memory,
        MAX(timestamp) as newest_memory
    FROM gamelan_memories
    WHERE namespace = p_namespace
        AND (expires_at IS NULL OR expires_at > NOW());
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for active memories (not expired)
CREATE OR REPLACE VIEW active_memories AS
SELECT 
    id, namespace, content, type, metadata, 
    timestamp, importance, tenant_id
FROM gamelan_memories
WHERE expires_at IS NULL OR expires_at > NOW();

-- View for memory statistics by namespace
CREATE OR REPLACE VIEW memory_stats_by_namespace AS
SELECT 
    namespace,
    tenant_id,
    COUNT(*) as total_memories,
    COUNT(*) FILTER (WHERE type = 'EPISODIC') as episodic_count,
    COUNT(*) FILTER (WHERE type = 'SEMANTIC') as semantic_count,
    AVG(importance) as avg_importance,
    MIN(timestamp) as oldest_memory,
    MAX(timestamp) as newest_memory
FROM gamelan_memories
WHERE expires_at IS NULL OR expires_at > NOW()
GROUP BY namespace, tenant_id;

-- ============================================================================
-- SAMPLE DATA (Optional, for testing)
-- ============================================================================

-- Uncomment to insert sample data
/*
INSERT INTO gamelan_memories (
    namespace, content, embedding, type, metadata, importance, tenant_id
) VALUES (
    'test:sample',
    'This is a sample memory for testing purposes.',
    '[0.1, 0.2, 0.3]'::vector,
    'EPISODIC',
    '{"category": "test", "priority": "low"}'::jsonb,
    0.5,
    'test-tenant'
);
*/

-- ============================================================================
-- GRANTS (Adjust as needed for your security model)
-- ============================================================================

-- Grant permissions to gamelan user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA gamelan TO gamelan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA gamelan TO gamelan;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA gamelan TO gamelan;

-- ============================================================================
-- MAINTENANCE
-- ============================================================================

-- Schedule periodic cleanup (requires pg_cron extension)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('clean-expired-memories', '0 2 * * *', 'SELECT clean_expired_memories()');

-- Analyze tables for query optimization
ANALYZE gamelan_memories;

-- ============================================================================
-- MONITORING QUERIES
-- ============================================================================

-- Example: Check index usage
-- SELECT 
--     schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
-- FROM pg_stat_user_indexes
-- WHERE schemaname = 'gamelan'
-- ORDER BY idx_scan DESC;

-- Example: Check table size
-- SELECT 
--     pg_size_pretty(pg_total_relation_size('gamelan.gamelan_memories')) as total_size,
--     pg_size_pretty(pg_relation_size('gamelan.gamelan_memories')) as table_size,
--     pg_size_pretty(pg_indexes_size('gamelan.gamelan_memories')) as indexes_size;

-- ============================================================================
-- COMPLETION
-- ============================================================================

\echo 'Gamelan Memory Store database initialized successfully!'
\echo 'pgvector extension: ENABLED'
\echo 'Schema: gamelan'
\echo 'Table: gamelan_memories'
\echo 'Indexes: 10 created'
\echo 'Functions: 3 created'
\echo 'Views: 2 created'

----------------------------------------------

package tech.kayys.gamelan.core.engine;

import tech.kayys.gamelan.core.domain.*;

import java.util.Map;

/**
 * ============================================================================
 * ENGINE TASK AND RESULT MODELS
 * ============================================================================
 */

// ==================== NODE EXECUTION TASK ====================

/**
 * Task sent to executor for execution
 */
public record NodeExecutionTask(
    WorkflowRunId runId,
    NodeId nodeId,
    int attempt,
    ExecutionToken token,
    Map<String, Object> context
) {
    /**
     * Get value from context
     */
    public Object getContextValue(String key) {
        return context.get(key);
    }
    
    /**
     * Get value with default
     */
    @SuppressWarnings("unchecked")
    public <T> T getContextValue(String key, T defaultValue) {
        return (T) context.getOrDefault(key, defaultValue);
    }
}

// ==================== NODE EXECUTION RESULT ====================

/**
 * Result from executor after task execution
 */
public record NodeExecutionResult(
    WorkflowRunId runId,
    NodeId nodeId,
    int attempt,
    NodeExecutionStatus status,
    Map<String, Object> output,
    ErrorInfo error,
    ExecutionToken executionToken
) {
    /**
     * Create successful result
     */
    public static NodeExecutionResult success(
            WorkflowRunId runId,
            NodeId nodeId,
            int attempt,
            Map<String, Object> output,
            ExecutionToken token) {
        
        return new NodeExecutionResult(
            runId,
            nodeId,
            attempt,
            NodeExecutionStatus.COMPLETED,
            output,
            null,
            token
        );
    }
    
    /**
     * Create failure result
     */
    public static NodeExecutionResult failure(
            WorkflowRunId runId,
            NodeId nodeId,
            int attempt,
            ErrorInfo error,
            ExecutionToken token) {
        
        return new NodeExecutionResult(
            runId,
            nodeId,
            attempt,
            NodeExecutionStatus.FAILED,
            Map.of(),
            error,
            token
        );
    }
    
    /**
     * Check if successful
     */
    public boolean isSuccess() {
        return status == NodeExecutionStatus.COMPLETED;
    }
    
    /**
     * Check if failed
     */
    public boolean isFailed() {
        return status == NodeExecutionStatus.FAILED;
    }
}

---------
package tech.kayys.gamelan.executor.memory;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import tech.kayys.gamelan.core.domain.*;
import tech.kayys.gamelan.core.engine.NodeExecutionResult;
import tech.kayys.gamelan.core.engine.NodeExecutionTask;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ============================================================================
 * MEMORY EXECUTOR TESTS
 * ============================================================================
 */
@QuarkusTest
public class MemoryExecutorTest {
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    ContextEngineeringService contextService;
    
    @Test
    public void testStoreAndRetrieveMemory() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        String content = "Test memory content";
        
        String memoryId = embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("test")
                    .content(content)
                    .embedding(embedding)
                    .type(MemoryType.EPISODIC)
                    .importance(0.8)
                    .build();
                
                return memoryStore.store(memory);
            })
            .await().indefinitely();
        
        assertNotNull(memoryId);
        
        Memory retrieved = memoryStore.retrieve(memoryId)
            .await().indefinitely();
        
        assertNotNull(retrieved);
        assertEquals(content, retrieved.getContent());
        assertEquals(MemoryType.EPISODIC, retrieved.getType());
        assertEquals(0.8, retrieved.getImportance(), 0.01);
    }
    
    @Test
    public void testSemanticSearch() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // Store some test memories
        List<String> contents = List.of(
            "Customer had issue with product quality",
            "Payment processing failed due to network error",
            "User requested refund for damaged item"
        );
        
        Uni.join().all(
            contents.stream()
                .map(content -> storeTestMemory(content, embeddingService))
                .toList()
        ).andFailFast()
        .await().indefinitely();
        
        // Search for similar
        String query = "Product quality complaint";
        
        List<ScoredMemory> results = embeddingService.embed(query)
            .flatMap(queryEmbedding -> 
                memoryStore.search(
                    queryEmbedding,
                    3,
                    0.0,
                    Map.of("namespace", "test")
                )
            )
            .await().indefinitely();
        
        assertNotNull(results);
        assertFalse(results.isEmpty());
        
        // First result should be most similar
        assertTrue(results.get(0).getScore() > 0);
    }
    
    @Test
    public void testContextEngineering() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        // Store knowledge base
        List<String> knowledge = List.of(
            "To process refund: verify order, check policy, execute payment reversal",
            "Standard refund policy allows returns within 30 days",
            "Premium customers get expedited refund processing"
        );
        
        Uni.join().all(
            knowledge.stream()
                .map(k -> storeTestMemory(k, embeddingService))
                .toList()
        ).andFailFast()
        .await().indefinitely();
        
        // Build context
        String query = "How to handle refund request?";
        
        ContextConfig config = ContextConfig.builder()
            .maxMemories(3)
            .systemPrompt("You are a support assistant")
            .includeMetadata(false)
            .build();
        
        EngineerContext context = contextService
            .buildContext(query, "test", config)
            .await().indefinitely();
        
        assertNotNull(context);
        assertTrue(context.getTotalTokens() > 0);
        assertFalse(context.getSections().isEmpty());
        
        String prompt = context.toPrompt();
        assertNotNull(prompt);
        assertTrue(prompt.contains(query));
    }
    
    @Test
    public void testMemoryMetadata() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        Memory memory = embeddingService.embed("Test with metadata")
            .map(embedding -> 
                Memory.builder()
                    .namespace("test")
                    .content("Test with metadata")
                    .embedding(embedding)
                    .type(MemoryType.SEMANTIC)
                    .importance(0.7)
                    .addMetadata("category", "testing")
                    .addMetadata("priority", "high")
                    .build()
            )
            .await().indefinitely();
        
        String memoryId = memoryStore.store(memory)
            .await().indefinitely();
        
        // Update metadata
        Map<String, Object> newMetadata = Map.of("status", "processed");
        
        Memory updated = memoryStore.updateMetadata(memoryId, newMetadata)
            .await().indefinitely();
        
        assertNotNull(updated);
        assertEquals("processed", updated.getMetadata().get("status"));
        assertEquals("testing", updated.getMetadata().get("category"));
    }
    
    @Test
    public void testMemoryStatistics() {
        // Store different types of memories
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        Uni.join().all(
            storeTestMemory("Episodic 1", MemoryType.EPISODIC, embeddingService),
            storeTestMemory("Episodic 2", MemoryType.EPISODIC, embeddingService),
            storeTestMemory("Semantic 1", MemoryType.SEMANTIC, embeddingService),
            storeTestMemory("Working 1", MemoryType.WORKING, embeddingService)
        ).andFailFast()
        .await().indefinitely();
        
        MemoryStatistics stats = memoryStore.getStatistics("test")
            .await().indefinitely();
        
        assertNotNull(stats);
        assertTrue(stats.getTotalMemories() >= 4);
        assertTrue(stats.getEpisodicCount() >= 2);
        assertTrue(stats.getSemanticCount() >= 1);
        assertTrue(stats.getWorkingCount() >= 1);
    }
    
    // Helper methods
    
    private Uni<String> storeTestMemory(String content, EmbeddingService embeddingService) {
        return storeTestMemory(content, MemoryType.EPISODIC, embeddingService);
    }
    
    private Uni<String> storeTestMemory(
            String content,
            MemoryType type,
            EmbeddingService embeddingService) {
        
        return embeddingService.embed(content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace("test")
                    .content(content)
                    .embedding(embedding)
                    .type(type)
                    .importance(0.6)
                    .build();
                
                return memoryStore.store(memory);
            });
    }
}




---------------------
package tech.kayys.gamelan.executor.memory;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ============================================================================
 * GAMELAN MEMORY EXECUTOR - MAIN APPLICATION
 * ============================================================================
 */
@QuarkusMain
public class MemoryExecutorApplication implements QuarkusApplication {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorApplication.class);
    
    public static void main(String[] args) {
        LOG.info("Starting Gamelan Memory Executor...");
        Quarkus.run(MemoryExecutorApplication.class, args);
    }
    
    @Override
    public int run(String... args) throws Exception {
        LOG.info("=".repeat(60));
        LOG.info("  Gamelan Memory Executor Started");
        LOG.info("  Version: 1.0.0");
        LOG.info("  Ready to process memory-aware workflow tasks");
        LOG.info("=".repeat(60));
        
        Quarkus.waitForExit();
        return 0;
    }
}

/**
 * Startup initializer
 */
@ApplicationScoped
class MemoryExecutorInitializer {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorInitializer.class);
    
    @Inject
    VectorStoreFactory vectorStoreFactory;
    
    @Inject
    EmbeddingServiceFactory embeddingServiceFactory;
    
    void onStart(@jakarta.enterprise.event.Observes io.quarkus.runtime.StartupEvent event) {
        LOG.info("Initializing Memory Executor components...");
        
        // Initialize vector store
        VectorMemoryStore vectorStore = vectorStoreFactory.getVectorStore();
        LOG.info("Vector store initialized: {}", vectorStore.getClass().getSimpleName());
        
        // Initialize embedding service
        EmbeddingService embeddingService = embeddingServiceFactory.getEmbeddingService();
        LOG.info("Embedding service initialized: {} (dimension: {})",
            embeddingService.getProvider(),
            embeddingService.getDimension());
        
        LOG.info("Memory Executor initialization complete");
    }
}


--------
package tech.kayys.gamelan.executor.memory.api;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import tech.kayys.gamelan.executor.memory.*;

import java.util.List;
import java.util.Map;

/**
 * ============================================================================
 * MEMORY EXECUTOR REST API
 * ============================================================================
 */
@Path("/api/memory")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class MemoryResource {
    
    @Inject
    VectorMemoryStore memoryStore;
    
    @Inject
    EmbeddingServiceFactory embeddingFactory;
    
    @Inject
    ContextEngineeringService contextService;
    
    /**
     * Store a new memory
     */
    @POST
    @Path("/store")
    public Uni<StoreResponse> storeMemory(StoreRequest request) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        return embeddingService.embed(request.content)
            .flatMap(embedding -> {
                Memory memory = Memory.builder()
                    .namespace(request.namespace != null ? request.namespace : "default")
                    .content(request.content)
                    .embedding(embedding)
                    .type(request.type != null ? request.type : MemoryType.EPISODIC)
                    .importance(request.importance != null ? request.importance : 0.5)
                    .metadata(request.metadata != null ? request.metadata : Map.of())
                    .build();
                
                return memoryStore.store(memory);
            })
            .map(memoryId -> new StoreResponse(true, memoryId, "Memory stored successfully"));
    }
    
    /**
     * Search for similar memories
     */
    @POST
    @Path("/search")
    public Uni<SearchResponse> search(SearchRequest request) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();
        
        return embeddingService.embed(request.query)
            .flatMap(queryEmbedding -> 
                memoryStore.search(
                    queryEmbedding,
                    request.limit != null ? request.limit : 10,
                    request.minSimilarity != null ? request.minSimilarity : 0.5,
                    Map.of("namespace", request.namespace != null ? request.namespace : "default")
                )
            )
            .map(results -> {
                List<MemoryResult> memoryResults = results.stream()
                    .map(scored -> new MemoryResult(
                        scored.getMemory().getId(),
                        scored.getMemory().getContent(),
                        scored.getScore(),
                        scored.getMemory().getType().name(),
                        scored.getMemory().getImportance()
                    ))
                    .toList();
                
                return new SearchResponse(true, memoryResults, memoryResults.size());
            });
    }
    
    /**
     * Build context for a query
     */
    @POST
    @Path("/context")
    public Uni<ContextResponse> buildContext(ContextRequest request) {
        ContextConfig config = ContextConfig.builder()
            .maxMemories(request.maxMemories != null ? request.maxMemories : 10)
            .systemPrompt(request.systemPrompt)
            .taskInstructions(request.taskInstructions)
            .includeMetadata(request.includeMetadata != null ? request.includeMetadata : true)
            .build();
        
        return contextService.buildContext(
                request.query,
                request.namespace != null ? request.namespace : "default",
                config
            )
            .map(context -> new ContextResponse(
                true,
                context.toPrompt(),
                context.getTotalTokens(),
                context.getUtilization(),
                context.getSections().size()
            ));
    }
    
    /**
     * Get memory statistics
     */
    @GET
    @Path("/stats/{namespace}")
    public Uni<StatsResponse> getStatistics(@PathParam("namespace") String namespace) {
        return memoryStore.getStatistics(namespace)
            .map(stats -> new StatsResponse(
                true,
                stats.getTotalMemories(),
                stats.getEpisodicCount(),
                stats.getSemanticCount(),
                stats.getProceduralCount(),
                stats.getWorkingCount(),
                stats.getAvgImportance()
            ));
    }
    
    /**
     * Run examples
     */
    @GET
    @Path("/examples/run")
    public Uni<ExampleResponse> runExamples() {
        // This would inject and run the examples
        return Uni.createFrom().item(
            new ExampleResponse(true, "Examples would run here. Check logs.", 7)
        );
    }
}

// ==================== REQUEST/RESPONSE MODELS ====================

record StoreRequest(
    String namespace,
    String content,
    MemoryType type,
    Double importance,
    Map<String, Object> metadata
) {}

record StoreResponse(
    boolean success,
    String memoryId,
    String message
) {}

record SearchRequest(
    String namespace,
    String query,
    Integer limit,
    Double minSimilarity
) {}

record SearchResponse(
    boolean success,
    List<MemoryResult> results,
    int count
) {}

record MemoryResult(
    String id,
    String content,
    double score,
    String type,
    double importance
) {}

record ContextRequest(
    String namespace,
    String query,
    Integer maxMemories,
    String systemPrompt,
    String taskInstructions,
    Boolean includeMetadata
) {}

record ContextResponse(
    boolean success,
    String prompt,
    int totalTokens,
    double utilization,
    int sectionCount
) {}

record StatsResponse(
    boolean success,
    long totalMemories,
    long episodicCount,
    long semanticCount,
    long proceduralCount,
    long workingCount,
    double avgImportance
) {}

record ExampleResponse(
    boolean success,
    String message,
    int examplesCount
) {}

--------