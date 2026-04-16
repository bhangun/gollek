package tech.kayys.gollek.plugin.cache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.embedding.EmbeddingService;
import tech.kayys.gollek.spi.embedding.EmbeddingService.EmbeddingResult;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.Message;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Semantic cache service that uses embedding similarity to cache and retrieve responses.
 *
 * <p>
 * Features:
 * <ul>
 * <li>Embedding-based similarity matching</li>
 * <li>Configurable similarity threshold</li>
 * <li>Automatic cache eviction (time-based and size-based)</li>
 * <li>Cache statistics and metrics</li>
 * <li>Thread-safe operations</li>
 * </ul>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class SemanticCacheService {

    private static final Logger LOG = Logger.getLogger(SemanticCacheService.class);

    @Inject
    EmbeddingService embeddingService;

    // Cache configuration
    private int maxCacheSize = 10000;
    private Duration expireAfterWrite = Duration.ofHours(24);
    private double similarityThreshold = 0.85;
    private int embeddingDimensions = 384; // Default for all-MiniLM-L6-v2

    // Actual cache: key -> cache entry
    private Cache<String, CacheEntry> cache;

    // Index for similarity search: stores embeddings for each key
    private final Map<String, float[]> embeddingIndex = new ConcurrentHashMap<>();

    // Statistics
    private final CacheStats stats = new CacheStats();

    /**
     * Cache entry containing the response and metadata.
     */
    public record CacheEntry(
            String key,
            String requestText,
            InferenceResponse response,
            float[] embedding,
            long createdAt,
            int hitCount) {

        public CacheEntry {
            Objects.requireNonNull(key);
            Objects.requireNonNull(requestText);
            Objects.requireNonNull(response);
            Objects.requireNonNull(embedding);
        }
    }

    /**
     * Cache statistics.
     */
    public static class CacheStats {
        private long hits = 0;
        private long misses = 0;
        private long evictions = 0;
        private long size = 0;

        public synchronized void recordHit() {
            hits++;
        }

        public synchronized void recordMiss() {
            misses++;
        }

        public synchronized void recordEviction() {
            evictions++;
        }

        public synchronized void updateSize(long size) {
            this.size = size;
        }

        public synchronized long getHits() {
            return hits;
        }

        public synchronized long getMisses() {
            return misses;
        }

        public synchronized long getEvictions() {
            return evictions;
        }

        public synchronized long getSize() {
            return size;
        }

        public synchronized double getHitRate() {
            long total = hits + misses;
            return total == 0 ? 0.0 : (double) hits / total;
        }

        @Override
        public synchronized String toString() {
            return String.format(
                "CacheStats[hits=%d, misses=%d, hitRate=%.2f%%, size=%d, evictions=%d]",
                hits, misses, getHitRate() * 100, size, evictions);
        }
    }

    @PostConstruct
    synchronized void init() {
        LOG.info("Initializing SemanticCacheService");

        this.cache = Caffeine.newBuilder()
                .maximumSize(maxCacheSize)
                .expireAfterWrite(expireAfterWrite.toSeconds(), TimeUnit.SECONDS)
                .removalListener((key, value, cause) -> {
                    if (value instanceof CacheEntry entry) {
                        embeddingIndex.remove(key);
                        stats.recordEviction();
                        LOG.debugf("Evicted cache entry: %s (reason: %s)", key, cause);
                    }
                })
                .recordStats()
                .build();

        LOG.infof("SemanticCacheService initialized with max size=%d, TTL=%s, threshold=%.2f",
                maxCacheSize, expireAfterWrite, similarityThreshold);
    }

    @PreDestroy
    void destroy() {
        if (cache != null) {
            cache.invalidateAll();
            embeddingIndex.clear();
            LOG.info("SemanticCacheService destroyed");
        }
    }

    /**
     * Check if a semantically similar request exists in cache.
     *
     * @param request the inference request
     * @return optional cached response
     */
    public Optional<InferenceResponse> get(InferenceRequest request) {
        String prompt = extractPrompt(request);

        LOG.debugf("Checking semantic cache for prompt: %s", prompt.substring(0, Math.min(50, prompt.length())));

        try {
            // Generate embedding for the request
            float[] queryEmbedding = generateEmbedding(prompt);

            // Find most similar entry in cache
            Optional<CacheEntry> bestMatch = findMostSimilar(queryEmbedding);

            if (bestMatch.isPresent()) {
                CacheEntry entry = bestMatch.get();
                double similarity = cosineSimilarity(queryEmbedding, entry.embedding);

                LOG.debugf("Found cache match with similarity: %.4f", similarity);

                if (similarity >= similarityThreshold) {
                    stats.recordHit();
                    updateStats();
                    LOG.infof("Cache HIT (similarity=%.4f)", similarity);
                    return Optional.of(entry.response);
                }
            }

            stats.recordMiss();
            updateStats();
            return Optional.empty();

        } catch (Exception e) {
            LOG.warnf(e, "Failed to check semantic cache, proceeding without cache");
            stats.recordMiss();
            return Optional.empty();
        }
    }

    /**
     * Store a request-response pair in the cache.
     *
     * @param request the inference request
     * @param response the inference response
     */
    public synchronized void put(InferenceRequest request, InferenceResponse response) {
        String prompt = extractPrompt(request);
        String key = generateKey(prompt);

        try {
            // Generate embedding
            float[] embedding = generateEmbedding(prompt);

            // Create cache entry
            CacheEntry entry = new CacheEntry(
                    key,
                    prompt,
                    response,
                    embedding,
                    System.currentTimeMillis(),
                    0);

            // Store in cache and index
            cache.put(key, entry);
            embeddingIndex.put(key, embedding);

            updateStats();

            LOG.debugf("Cached response for key: %s", key);

        } catch (Exception e) {
            LOG.warnf(e, "Failed to cache response for key: %s", key);
        }
    }

    /**
     * Remove a specific entry from cache.
     *
     * @param key the cache key
     */
    public synchronized void remove(String key) {
        cache.invalidate(key);
        embeddingIndex.remove(key);
        updateStats();
        LOG.debugf("Removed cache entry: %s", key);
    }

    /**
     * Clear all cache entries.
     */
    public synchronized void clear() {
        cache.invalidateAll();
        embeddingIndex.clear();
        updateStats();
        LOG.info("Semantic cache cleared");
    }

    /**
     * Get cache statistics.
     *
     * @return cache statistics
     */
    public CacheStats getStats() {
        updateStats();
        return stats;
    }

    /**
     * Get all cache keys.
     *
     * @return set of cache keys
     */
    public Set<String> getKeys() {
        return Set.copyOf(embeddingIndex.keySet());
    }

    /**
     * Get number of entries in cache.
     *
     * @return cache size
     */
    public long size() {
        return cache.estimatedSize();
    }

    // ═══════════════════════════════════════════════════════════════
    // Configuration Methods
    // ═══════════════════════════════════════════════════════════════

    public void setMaxCacheSize(int maxCacheSize) {
        this.maxCacheSize = maxCacheSize;
        LOG.infof("Updated max cache size: %d", maxCacheSize);
    }

    public void setExpireAfterWrite(Duration duration) {
        this.expireAfterWrite = duration;
        LOG.infof("Updated cache TTL: %s", duration);
    }

    public void setSimilarityThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("Similarity threshold must be between 0.0 and 1.0");
        }
        this.similarityThreshold = threshold;
        LOG.infof("Updated similarity threshold: %.2f", threshold);
    }

    public void setEmbeddingDimensions(int dimensions) {
        this.embeddingDimensions = dimensions;
        LOG.infof("Updated embedding dimensions: %d", dimensions);
    }

    // ═══════════════════════════════════════════════════════════════
    // Private Helper Methods
    // ═══════════════════════════════════════════════════════════════

    private String extractPrompt(InferenceRequest request) {
        // Extract the main prompt/text from the request
        if (request.getPrompt() != null && !request.getPrompt().isBlank()) {
            return request.getPrompt();
        }

        // Fallback: concatenate messages if present
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (var msg : request.getMessages()) {
                if (msg.getContent() != null) {
                    sb.append(msg.getContent()).append(" ");
                }
            }
            return sb.toString().trim();
        }

        return "";
    }

    private String generateKey(String prompt) {
        // Generate a unique key based on prompt hash
        return "cache_" + Integer.toHexString(prompt.hashCode());
    }

    private float[] generateEmbedding(String text) {
        if (text == null || text.isBlank()) {
            return new float[embeddingDimensions];
        }

        try {
            MultimodalContent content = MultimodalContent.ofText(text);
            
            EmbeddingResult result = embeddingService.embed(content, "all-MiniLM-L6-v2")
                .await().atMost(Duration.ofSeconds(10));

            if (result != null && result.vector() != null) {
                return result.vector();
            }

            // Fallback: use simple hash-based embedding
            return generateSimpleEmbedding(text);

        } catch (Exception e) {
            LOG.warnf(e, "Embedding generation failed, using fallback");
            return generateSimpleEmbedding(text);
        }
    }

    private float[] generateSimpleEmbedding(String text) {
        // Simple hash-based embedding (fallback)
        float[] embedding = new float[embeddingDimensions];
        char[] chars = text.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            int idx = i % embeddingDimensions;
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

    private Optional<CacheEntry> findMostSimilar(float[] queryEmbedding) {
        if (embeddingIndex.isEmpty()) {
            return Optional.empty();
        }

        String bestKey = null;
        double bestSimilarity = -1.0;

        for (Map.Entry<String, float[]> entry : embeddingIndex.entrySet()) {
            double similarity = cosineSimilarity(queryEmbedding, entry.getValue());

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestKey = entry.getKey();
            }
        }

        if (bestKey != null && bestSimilarity > 0) {
            return Optional.ofNullable(cache.getIfPresent(bestKey));
        }

        return Optional.empty();
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

    private void updateStats() {
        stats.updateSize(cache.estimatedSize());
    }
}
