package tech.kayys.gollek.server;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Application-scoped holder for a GollekSdk instance.
 * Keeps a single SDK instance for request handlers to use.
 *
 * Falls back to an in-process dummy SDK when no provider is available
 * — useful for demos and tests.
 */
@ApplicationScoped
public class SdkProvider {

    private static final Logger LOG = Logger.getLogger(SdkProvider.class);

    private GollekSdk sdk;

    @Inject
    @ConfigProperty(name = "gollek.server.allowed-api-keys", defaultValue = "community")
    String allowedApiKeys;

    @PostConstruct
    void init() {
        try {
            this.sdk = GollekSdkFactory.createLocalSdk();
        } catch (SdkException e) {
            LOG.warn("No local Gollek SDK found on classpath; using demo fallback. " + e.getMessage());
            this.sdk = new DemoSdk();
        }
    }

    public GollekSdk getSdk() {
        return sdk;
    }

    /**
     * Simple demo SDK used when no real provider is available. Implements a
     * small subset of the GollekSdk API sufficient for demos and tests.
     */
    private static class DemoSdk implements GollekSdk {

        @Override
        public tech.kayys.gollek.spi.inference.InferenceResponse createCompletion(InferenceRequest request) {
            return new InferenceResponse.Builder()
                    .requestId(request.getRequestId())
                    .content("[demo] echo: " + (request.getPrompt() != null ? request.getPrompt() : ""))
                    .model(request.getModel())
                    .build();
        }

        @Override
        public java.util.concurrent.CompletableFuture<tech.kayys.gollek.spi.inference.InferenceResponse> createCompletionAsync(InferenceRequest request) {
            return java.util.concurrent.CompletableFuture.completedFuture(createCompletion(request));
        }

        @Override
        public io.smallrye.mutiny.Multi<tech.kayys.gollek.spi.inference.StreamingInferenceChunk> streamCompletion(InferenceRequest request) {
            return io.smallrye.mutiny.Multi.createFrom().empty();
        }

        @Override
        public tech.kayys.gollek.spi.embedding.EmbeddingResponse createEmbedding(EmbeddingRequest request) {
            float[] v = new float[16];
            for (int i = 0; i < v.length; i++) v[i] = 0.0f;
            return new EmbeddingResponse(request.requestId(), request.model(), java.util.List.of(v), v.length, Map.of());
        }

        @Override
        public String submitAsyncJob(InferenceRequest request) {
            return "demo-job-" + java.util.UUID.randomUUID();
        }

        @Override
        public tech.kayys.gollek.spi.inference.AsyncJobStatus getJobStatus(String jobId) {
            return new tech.kayys.gollek.spi.inference.AsyncJobStatus(jobId, "COMPLETED", null);
        }

        @Override
        public tech.kayys.gollek.spi.inference.AsyncJobStatus waitForJob(String jobId, java.time.Duration maxWaitTime, java.time.Duration pollInterval) {
            return getJobStatus(jobId);
        }

        @Override
        public java.util.List<tech.kayys.gollek.spi.inference.InferenceResponse> batchInference(tech.kayys.gollek.spi.batch.BatchInferenceRequest batchRequest) {
            return java.util.List.of();
        }

        @Override
        public java.util.List<tech.kayys.gollek.spi.provider.ProviderInfo> listAvailableProviders() {
            return java.util.List.of();
        }

        @Override
        public tech.kayys.gollek.spi.provider.ProviderInfo getProviderInfo(String providerId) {
            return null;
        }

        @Override
        public void setPreferredProvider(String providerId) {
        }

        @Override
        public Optional<String> getPreferredProvider() {
            return Optional.empty();
        }

        @Override
        public java.util.List<tech.kayys.gollek.sdk.model.ModelInfo> listModels() {
            return java.util.List.of();
        }

        @Override
        public java.util.List<tech.kayys.gollek.sdk.model.ModelInfo> listModels(int offset, int limit) {
            return java.util.List.of();
        }

        @Override
        public Optional<tech.kayys.gollek.sdk.model.ModelInfo> getModelInfo(String modelId) {
            return Optional.empty();
        }

        @Override
        public void pullModel(String modelSpec, java.util.function.Consumer<tech.kayys.gollek.sdk.model.PullProgress> progressCallback) {
            // simulate progress
            for (int i = 1; i <= 5; i++) {
                progressCallback.accept(new tech.kayys.gollek.sdk.model.PullProgress(i * 20, "step " + i));
                try { Thread.sleep(50); } catch (InterruptedException ignored) {}
            }
        }

        @Override
        public void deleteModel(String modelId) {
        }

        @Override
        public void pullModel(String modelSpec, String revision, boolean force, java.util.function.Consumer<tech.kayys.gollek.sdk.model.PullProgress> progressCallback) {
            pullModel(modelSpec, progressCallback);
        }

        @Override
        public tech.kayys.gollek.sdk.model.ModelResolution prepareModel(String modelId, boolean forceGguf, java.util.function.Consumer<tech.kayys.gollek.sdk.model.PullProgress> progressCallback) {
            return new tech.kayys.gollek.sdk.model.ModelResolution(modelId, null, null);
        }

        @Override
        public Optional<String> autoSelectProvider(String modelId, boolean forceGguf) {
            return Optional.empty();
        }

        @Override
        public tech.kayys.gollek.sdk.model.SystemInfo getSystemInfo() {
            return new tech.kayys.gollek.sdk.model.SystemInfo(java.util.Map.of(), java.util.Map.of());
        }

        // other default methods remain unimplemented for brevity
    }
}
