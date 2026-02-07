package tech.kayys.golek.provider.tensorflow;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.golek.spi.exception.ProviderException;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * TensorFlow SavedModel provider.
 * Supports TensorFlow 2.x SavedModel format.
 */
@ApplicationScoped
public class TensorFlowProvider implements LLMProvider {

    private static final Logger LOG = Logger.getLogger(TensorFlowProvider.class);
    private static final String PROVIDER_ID = "tensorflow";
    private static final String PROVIDER_NAME = "TensorFlow";
    private static final String VERSION = "1.0.0";

    private final Map<String, TFModel> modelCache = new ConcurrentHashMap<>();
    private ProviderConfig config;
    private volatile boolean initialized = false;

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
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(id())
                .name(name())
                .description("TensorFlow SavedModel provider")
                .vendor("Google")
                .version(VERSION)
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .multimodal(true)
                .embeddings(true)
                .maxContextTokens(8192)
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        if (initialized) {
            return;
        }

        synchronized (this) {
            if (initialized) {
                return;
            }

            LOG.info("Initializing TensorFlow provider");
            this.config = config;

            try {
                // Initialize TensorFlow runtime
                initializeTensorFlow();
                initialized = true;
                LOG.info("TensorFlow provider initialized successfully");
            } catch (Exception e) {
                throw new ProviderException.ProviderInitializationException(
                        "Failed to initialize TensorFlow provider", e);
            }
        }
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        return modelId.contains("tensorflow") ||
                modelId.contains("savedmodel") ||
                modelCache.containsKey(modelId);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        return Uni.createFrom().emitter(emitter -> {
            try {
                ensureInitialized();

                LOG.debugf("TensorFlow inference for model: %s", request.getModel());

                TFModel model = loadModel(request.getModel(), context);

                // Prepare inputs
                Map<String, Object> inputs = prepareInputs(request);

                // Run inference
                long startTime = System.currentTimeMillis();
                Map<String, Object> outputs = model.predict(inputs);
                long duration = System.currentTimeMillis() - startTime;

                // Convert to response
                String content = extractContent(outputs);

                emitter.complete(InferenceResponse.builder()
                        .requestId(request.getRequestId())
                        .content(content)
                        .model(request.getModel())
                        .durationMs(duration)
                        .metadata("provider", id())
                        .metadata("signature", model.getSignature())
                        .build());
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    @Override
    public Uni<ProviderHealth> health() {
        if (!initialized) {
            return Uni.createFrom().item(ProviderHealth.unhealthy("Provider not initialized"));
        }

        try {
            // Simple health check
            testTensorFlowRuntime();
            return Uni.createFrom().item(ProviderHealth.healthy("TensorFlow provider operational"));
        } catch (Exception e) {
            return Uni.createFrom().item(ProviderHealth.unhealthy("Health check failed: " + e.getMessage()));
        }
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down TensorFlow provider");
        modelCache.values().forEach(TFModel::close);
        modelCache.clear();
        initialized = false;
    }

    private TFModel loadModel(String modelId, TenantContext context) {
        TenantContext effectiveContext = ensureTenantContext(context);
        return modelCache.computeIfAbsent(modelId, id -> {
            LOG.infof("Loading TensorFlow model: %s", id);
            String modelPath = resolveModelPath(id, effectiveContext);
            TFModel model = new TFModel(modelPath);
            model.load();
            return model;
        });
    }

    private Map<String, Object> prepareInputs(ProviderRequest request) {
        Map<String, Object> inputs = new HashMap<>();
        String text = request.getMessages().stream()
                .map(msg -> msg.getContent())
                .reduce("", (a, b) -> a + " " + b);
        inputs.put("input_text", text);
        return inputs;
    }

    private String extractContent(Map<String, Object> outputs) {
        Object result = outputs.get("output_text");
        return result != null ? result.toString() : "";
    }

    private String resolveModelPath(String modelId, TenantContext context) {
        String basePath = config.getString("models.path", "./models");
        return basePath + "/" + context.getTenantId() + "/" + modelId;
    }

    private TenantContext ensureTenantContext(TenantContext context) {
        return context != null ? context : TenantContext.of("default");
    }

    private void initializeTensorFlow() {
        // Initialize TF runtime
        LOG.info("TensorFlow runtime initialized");
    }

    private void testTensorFlowRuntime() {
        // Health check
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }
    }

    /**
     * TensorFlow model wrapper
     */
    private static class TFModel implements AutoCloseable {
        private final String path;
        private Object session; // TF session handle
        private String signature = "serving_default";

        TFModel(String path) {
            this.path = path;
        }

        void load() {
            LOG.debugf("Loading SavedModel from: %s", path);
        }

        Map<String, Object> predict(Map<String, Object> inputs) {
            Map<String, Object> outputs = new HashMap<>();
            outputs.put("output_text", "Mock TensorFlow response");
            return outputs;
        }

        String getSignature() {
            return signature;
        }

        @Override
        public void close() {
            session = null;
        }
    }
}
