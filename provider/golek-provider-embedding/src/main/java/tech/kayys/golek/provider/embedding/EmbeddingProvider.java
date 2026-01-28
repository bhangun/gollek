package tech.kayys.golek.provider.embedding;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.provider.core.adapter.LocalProviderAdapter;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dedicated local embedding provider.
 * Simulates sentence-transformers execution.
 */
@ApplicationScoped
public class EmbeddingProvider extends LocalProviderAdapter {

    private static final Logger LOG = Logger.getLogger(EmbeddingProvider.class);
    private static final String PROVIDER_ID = "embedding-local";
    private static final String PROVIDER_NAME = "Local Embeddings";
    private static final String VERSION = "1.0.0";

    @Inject
    EmbeddingConfig config;

    @Inject
    ObjectMapper objectMapper;

    private final Map<String, Object> loadedModels = new ConcurrentHashMap<>();

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return PROVIDER_NAME;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .functionCalling(false)
                .multimodal(false)
                .embeddings(true)
                .maxContextTokens(512)
                .supportedModels(Set.of(
                        "all-MiniLM-L6-v2",
                        "bge-small-en-v1.5",
                        "e5-small-v2",
                        "paraphrase-multilingual-MiniLM-L12-v2"))
                .supportedFormats(Set.of("sentence-transformers", "onnx"))
                .metadata("batch_embedding", "true")
                .metadata("deployment", "local")
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .displayName(PROVIDER_NAME)
                .description("High-performance local embeddings using Sentence Transformers / ONNX")
                .version(VERSION)
                .vendor("Golek")
                .metadata("deployment", "local")
                .metadata("requires_api_key", "false")
                .build();
    }

    @Override
    public boolean supports(String model, TenantContext context) {
        return capabilities().getSupportedModels().contains(model) ||
                model.contains("embedding") ||
                model.contains("MiniLM") ||
                model.contains("bge") ||
                model.contains("e5");
    }

    @Override
    protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            long startTime = System.currentTimeMillis();

            // Extract texts
            List<String> texts = extractTexts(request);
            String modelId = request.getModel();

            LOG.debugf("Generating embeddings for %d texts using model %s", texts.size(), modelId);

            // Generate mock embeddings
            int dimension = config.defaultDimension();
            List<float[]> embeddings = generateMockEmbeddings(texts.size(), dimension);

            long duration = System.currentTimeMillis() - startTime;

            try {
                String content = objectMapper.writeValueAsString(embeddings);

                return InferenceResponse.builder()
                        .requestId(request.getRequestId())
                        .content(content)
                        .model(modelId)
                        .durationMs(duration)
                        .metadata("provider", PROVIDER_ID)
                        .metadata("embedding_dim", dimension)
                        .metadata("num_embeddings", embeddings.size())
                        .metadata("total_tokens", estimateTokens(texts))
                        .build();
            } catch (Exception e) {
                throw new RuntimeException("Failed to serialize embeddings", e);
            }
        });
    }

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant) {
        LOG.infof("Initializing Embedding provider for tenant: %s", tenant.getTenantId());
        return Uni.createFrom().voidItem();
    }

    @Override
    protected Uni<ProviderHealth> doHealthCheck() {
        return Uni.createFrom().item(() -> ProviderHealth.healthy(
                id(),
                Duration.ZERO,
                Map.of(
                        "loaded_models", loadedModels.size(),
                        "backend", "mock-cpu")));
    }

    @Override
    protected tech.kayys.golek.provider.core.loader.ModelLoader createModelLoader(Map<String, Object> config) {
        return new tech.kayys.golek.provider.core.loader.ModelLoader() {
            @Override
            public Uni<java.nio.file.Path> load(String modelId) {
                return Uni.createFrom().item(java.nio.file.Path.of(modelId));
            }

            @Override
            public boolean isLoaded(String modelId) {
                return true;
            }

            @Override
            public Uni<Void> unload(String modelId) {
                return Uni.createFrom().voidItem();
            }

            @Override
            public java.nio.file.Path getPath(String modelId) {
                return java.nio.file.Path.of(modelId);
            }
        };
    }

    @Override
    protected tech.kayys.golek.provider.core.session.SessionManager createSessionManager(Map<String, Object> config) {
        return new tech.kayys.golek.provider.core.session.SessionManager(
                tech.kayys.golek.provider.core.session.SessionConfig.defaults());
    }

    @Override
    protected boolean supportsStreamingInternally() {
        return false;
    }

    private List<String> extractTexts(ProviderRequest request) {
        List<String> texts = new ArrayList<>();
        if (request.getMessages() != null) {
            request.getMessages().forEach(msg -> texts.add(msg.getContent()));
        }
        return texts;
    }

    private List<float[]> generateMockEmbeddings(int count, int dimension) {
        List<float[]> embeddings = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            float[] vector = new float[dimension];
            double sum = 0;
            for (int j = 0; j < dimension; j++) {
                vector[j] = (float) (Math.random() * 2 - 1);
                sum += vector[j] * vector[j];
            }
            // Normalize
            float norm = (float) Math.sqrt(sum);
            for (int j = 0; j < dimension; j++) {
                vector[j] /= norm;
            }
            embeddings.add(vector);
        }
        return embeddings;
    }

    private int estimateTokens(List<String> texts) {
        return texts.stream().mapToInt(String::length).sum() / 4;
    }
}
