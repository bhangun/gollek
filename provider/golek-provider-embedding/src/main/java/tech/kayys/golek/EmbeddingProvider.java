package tech.kayys.golek.core.plugin;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.provider.core.exception.ProviderException.ProviderInitializationException;
import tech.kayys.golek.provider.core.spi.LLMProvider;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dedicated embedding model provider.
 * Supports sentence transformers and specialized embedding models.
 * 
 * Capabilities:
 * - Text embeddings
 * - Semantic search
 * - Batch embedding
 * - Multiple embedding dimensions
 */
@ApplicationScoped
public class EmbeddingProvider implements LLMProvider {

    private static final Logger LOG = Logger.getLogger(EmbeddingProvider.class);

    private final Map<String, EmbeddingModel> modelCache = new ConcurrentHashMap<>();
    private ProviderConfig config;
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return "tech.kayys/embedding-provider";
    }

    @Override
    public String name() {
        return "Embedding Provider";
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .functionCalling(false)
                .multimodal(false)
                .embeddings(true)
                .maxContextTokens(512)
                .supportedFormats(Set.of("sentence-transformers", "sbert"))
                .supportedDevices(Set.of("cpu", "cuda"))
                .features(Map.of(
                        "batch_embedding", true,
                        "normalized_embeddings", true,
                        "pooling_strategies", List.of("mean", "cls", "max")))
                .build();
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        return modelId.contains("embedding") ||
                modelId.contains("sentence-transformers") ||
                modelId.contains("e5") ||
                modelId.contains("bge");
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderInitializationException {
        if (initialized) {
            return;
        }

        synchronized (this) {
            if (initialized) {
                return;
            }

            LOG.info("Initializing Embedding provider");
            this.config = config;

            try {
                // Initialize embedding runtime
                initialized = true;
                LOG.info("Embedding provider initialized");

            } catch (Exception e) {
                throw new ProviderInitializationException(
                        "Failed to initialize embedding provider", e);
            }
        }
    }

    @Override
    public Uni<InferenceResponse> infer(
            ProviderRequest request,
            TenantContext context) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();

            String modelId = request.getModel();
            LOG.debugf("Embedding inference for model: %s", modelId);

            EmbeddingModel model = loadModel(modelId, context);

            // Extract texts to embed
            List<String> texts = extractTexts(request);

            // Generate embeddings
            long startTime = System.currentTimeMillis();
            List<float[]> embeddings = model.embed(texts);
            long duration = System.currentTimeMillis() - startTime;

            // Format response
            String content = formatEmbeddings(embeddings);

            return InferenceResponse.builder()
                    .requestId(request.getMetadata("request_id")
                            .orElse(UUID.randomUUID().toString()))
                    .content(content)
                    .model(modelId)
                    .tokensUsed(estimateTokens(texts))
                    .durationMs(duration)
                    .metadata("provider", id())
                    .metadata("embedding_dim", embeddings.get(0).length)
                    .metadata("num_embeddings", embeddings.size())
                    .build();
        });
    }

    @Override
    public ProviderHealth health() {
        if (!initialized) {
            return ProviderHealth.unhealthy("Provider not initialized");
        }

        return ProviderHealth.healthy(
                "Embedding provider operational",
                Map.of("loaded_models", modelCache.size()));
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down Embedding provider");

        modelCache.values().forEach(EmbeddingModel::close);
        modelCache.clear();

        initialized = false;
    }

    private EmbeddingModel loadModel(String modelId, TenantContext context) {
        return modelCache.computeIfAbsent(modelId, id -> {
            LOG.infof("Loading embedding model: %s", id);

            String modelPath = resolveModelPath(id, context);
            int dimension = config.getInt("embedding.dimension", 768);

            EmbeddingModel model = new EmbeddingModel(modelPath, dimension);
            model.load();

            return model;
        });
    }

    private List<String> extractTexts(ProviderRequest request) {
        List<String> texts = new ArrayList<>();

        request.getMessages().forEach(msg -> {
            texts.add(msg.getContent());
        });

        return texts;
    }

    private String formatEmbeddings(List<float[]> embeddings) {
        // Format as JSON array
        StringBuilder sb = new StringBuilder("[");

        for (int i = 0; i < embeddings.size(); i++) {
            if (i > 0)
                sb.append(",");

            sb.append("[");
            float[] embedding = embeddings.get(i);
            for (int j = 0; j < embedding.length; j++) {
                if (j > 0)
                    sb.append(",");
                sb.append(embedding[j]);
            }
            sb.append("]");
        }

        sb.append("]");
        return sb.toString();
    }

    private int estimateTokens(List<String> texts) {
        return texts.stream()
                .mapToInt(t -> t.split("\\s+").length)
                .sum();
    }

    private String resolveModelPath(String modelId, TenantContext context) {
        String basePath = config.getString("models.path", "./models");
        return basePath + "/" + context.getTenantId() + "/" + modelId;
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }
    }

    /**
     * Embedding model wrapper
     */
    private static class EmbeddingModel implements AutoCloseable {
        private final String path;
        private final int dimension;
        private Object modelHandle;

        EmbeddingModel(String path, int dimension) {
            this.path = path;
            this.dimension = dimension;
        }

        void load() {
            LOG.debugf("Loading embedding model from: %s", path);
            // Load model (simplified)
        }

        List<float[]> embed(List<String> texts) {
            // Generate embeddings
            // Simplified - use actual sentence transformers
            List<float[]> embeddings = new ArrayList<>();

            for (String text : texts) {
                float[] embedding = new float[dimension];
                // Fill with mock data
                for (int i = 0; i < dimension; i++) {
                    embedding[i] = (float) Math.random();
                }
                // Normalize
                normalize(embedding);
                embeddings.add(embedding);
            }

            return embeddings;
        }

        private void normalize(float[] vector) {
            double sum = 0.0;
            for (float v : vector) {
                sum += v * v;
            }
            double norm = Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }

        @Override
        public void close() {
            if (modelHandle != null) {
                // Release resources
                modelHandle = null;
            }
        }
    }
}