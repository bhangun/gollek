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
import java.util.Base64;

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

    @ConfigProperty(name = "gamelan.embedding.openai.endpoint", defaultValue = "https://api.openai.com/v1/embeddings")
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
                "float");

        return restClient.createEmbedding(
                "Bearer " + apiKey.orElseThrow(() -> new IllegalStateException("OpenAI API key not configured")),
                request)
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
                .onFailure().invoke(error -> LOG.error("Failed to generate embedding", error));
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
                "float");

        return restClient.createEmbeddingBatch(
                "Bearer " + apiKey.orElseThrow(() -> new IllegalStateException("OpenAI API key not configured")),
                request)
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