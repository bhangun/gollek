package tech.kayys.gollek.sdk.local;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.core.model.AsyncJobStatus;
import tech.kayys.gollek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.gollek.sdk.core.model.ModelInfo;
import tech.kayys.gollek.sdk.core.model.PullProgress;
import tech.kayys.gollek.sdk.core.mcp.McpRegistryManager;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Client-facing local SDK entrypoint for Gollek features.
 * Consumers should depend on/use this API instead of gollek-sdk-core directly.
 */
public interface GollekLocalClient {

    InferenceResponse createCompletion(InferenceRequest request);

    CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request);

    Multi<StreamChunk> streamCompletion(InferenceRequest request);

    String submitAsyncJob(InferenceRequest request);

    AsyncJobStatus getJobStatus(String jobId);

    AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval);

    List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest);

    List<ProviderInfo> listAvailableProviders();

    ProviderInfo getProviderInfo(String providerId);

    void setPreferredProvider(String providerId);

    Optional<String> getPreferredProvider();

    List<ModelInfo> listModels();

    List<ModelInfo> listModels(int offset, int limit);

    Optional<ModelInfo> getModelInfo(String modelId);

    void pullModel(String modelSpec, Consumer<PullProgress> progressCallback);

    void deleteModel(String modelId);

    McpRegistryManager mcpRegistry();
}
