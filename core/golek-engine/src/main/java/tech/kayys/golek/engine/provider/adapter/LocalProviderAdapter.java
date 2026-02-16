package tech.kayys.golek.engine.provider.adapter;

import io.smallrye.mutiny.Uni;

import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.engine.loader.ModelLoader;
import tech.kayys.golek.engine.session.SessionManager;

import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base adapter for local model execution (GGUF, ONNX, etc.)
 * Handles model loading, session management, and resource cleanup.
 */
public abstract class LocalProviderAdapter extends AbstractProvider {

    protected ModelLoader modelLoader;
    protected SessionManager sessionManager;
    protected final Map<String, Path> loadedModels = new ConcurrentHashMap<>();

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config) {
        return Uni.createFrom().item(() -> {
            this.modelLoader = createModelLoader(config);
            this.sessionManager = createSessionManager(config);

            log.infof("Local provider %s initialized with model loader and session manager",
                    id());

            return null;
        });
    }

    @Override
    protected Uni<ProviderHealth> doHealthCheck() {
        return Uni.createFrom().item(() -> {
            if (modelLoader == null || sessionManager == null) {
                return ProviderHealth.unhealthy("Provider components not initialized");
            }

            long activeModels = loadedModels.size();
            int activeSessions = sessionManager.activeSessionCount();

            return ProviderHealth.builder()
                    .status(ProviderHealth.Status.HEALTHY)
                    .message("Local provider operational")
                    .detail("loaded_models", activeModels)
                    .detail("active_sessions", activeSessions)
                    .build();
        });
    }

    @Override
    protected void doShutdown() {
        if (sessionManager != null) {
            sessionManager.shutdown();
        }

        loadedModels.clear();

        log.infof("Local provider %s shutdown complete", id());
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(supportsStreamingInternally())
                .functionCalling(false)
                .multimodal(false)
                .embeddings(false)
                .maxContextTokens(getMaxContextTokens())
                .maxOutputTokens(getMaxOutputTokens())
                .build();
    }

    /**
     * Load model if not already loaded
     */
    protected Uni<Path> ensureModelLoaded(String modelId) {
        Path cached = loadedModels.get(modelId);
        if (cached != null) {
            return Uni.createFrom().item(cached);
        }

        return modelLoader.load(modelId)
                .invoke(path -> {
                    loadedModels.put(modelId, path);
                    log.infof("Model %s loaded at %s", modelId, path);
                });
    }

    /**
     * Get model path
     */
    protected Optional<Path> getModelPath(String modelId) {
        return Optional.ofNullable(loadedModels.get(modelId));
    }

    /**
     * Create model loader instance
     */
    protected abstract ModelLoader createModelLoader(Map<String, Object> config);

    /**
     * Create session manager instance
     */
    protected abstract SessionManager createSessionManager(Map<String, Object> config);

    /**
     * Check if provider supports streaming internally
     */
    protected abstract boolean supportsStreamingInternally();

    /**
     * Get max context tokens supported
     */
    protected int getMaxContextTokens() {
        return getConfigValue("max-context-tokens", 4096);
    }

    /**
     * Get max output tokens supported
     */
    protected int getMaxOutputTokens() {
        return getConfigValue("max-output-tokens", 2048);
    }
}