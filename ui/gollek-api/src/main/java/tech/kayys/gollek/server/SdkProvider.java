package tech.kayys.gollek.server;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.time.Instant;

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

    @Produces
    @ApplicationScoped
    public GollekSdk produceSdk() {
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
            String content = "[demo] echo: " + (request.getPrompt() != null ? request.getPrompt() : "");
            return io.smallrye.mutiny.Multi.createFrom().items(
                    tech.kayys.gollek.spi.inference.StreamingInferenceChunk.of(request.getRequestId(), 0, content),
                    tech.kayys.gollek.spi.inference.StreamingInferenceChunk.finalTextChunk(
                            request.getRequestId(),
                            1,
                            "",
                            new tech.kayys.gollek.spi.inference.StreamingInferenceChunk.ChunkUsage(0, 0, 0)));
        }

        @Override
        public tech.kayys.gollek.spi.embedding.EmbeddingResponse createEmbedding(EmbeddingRequest request) {
            float[] v = new float[16];
            for (int i = 0; i < v.length; i++) v[i] = 0.0f;
            return new EmbeddingResponse(request.requestId(), request.model(), java.util.List.of(v), v.length, Map.of());
        }

        @Override
        public tech.kayys.gollek.spi.inference.AsyncJobStatus getJobStatus(String jobId) {
            Instant now = Instant.now();
            return new tech.kayys.gollek.spi.inference.AsyncJobStatus(jobId, null, "COMPLETED", null, null, now, now);
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
        public java.util.List<tech.kayys.gollek.spi.model.ModelInfo> listModels() {
            return java.util.List.of();
        }

        @Override
        public java.util.List<tech.kayys.gollek.spi.model.ModelInfo> listModels(int offset, int limit) {
            return java.util.List.of();
        }

        @Override
        public Optional<tech.kayys.gollek.spi.model.ModelInfo> getModelInfo(String modelId) {
            return Optional.empty();
        }

        @Override
        public String submitAsyncJob(InferenceRequest request) { return null; }

        @Override
        public void pullModel(String modelSpec, java.util.function.Consumer<tech.kayys.gollek.sdk.model.PullProgress> progressCallback) {}

        @Override
        public void deleteModel(String modelId) {}

        @Override
        public tech.kayys.gollek.sdk.model.ModelResolution prepareModel(String modelId, boolean forceGguf, java.util.function.Consumer<tech.kayys.gollek.sdk.model.PullProgress> progressCallback) {
            return new tech.kayys.gollek.sdk.model.ModelResolution(modelId, null, null);
        }

        @Override
        public tech.kayys.gollek.sdk.model.SystemInfo getSystemInfo() {
            Runtime runtime = Runtime.getRuntime();
            return tech.kayys.gollek.sdk.model.SystemInfo.builder()
                    .cliVersion("demo")
                    .javaVersion(System.getProperty("java.version"))
                    .osName(System.getProperty("os.name"))
                    .osVersion(System.getProperty("os.version"))
                    .osArch(System.getProperty("os.arch"))
                    .userName(System.getProperty("user.name"))
                    .userHome(System.getProperty("user.home"))
                    .totalMemory(runtime.totalMemory())
                    .freeMemory(runtime.freeMemory())
                    .maxMemory(runtime.maxMemory())
                    .availableProcessors(runtime.availableProcessors())
                    .metadata(Map.of("mode", "demo"))
                    .build();
        }
    }
}
