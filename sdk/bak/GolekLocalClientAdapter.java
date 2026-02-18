package tech.kayys.gollek.sdk.local;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.exception.SdkException;
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
 * CDI adapter exposing local SDK operations via a stable local client API.
 */
@ApplicationScoped
public class GollekLocalClientAdapter implements GollekLocalClient {

    @Inject
    GollekSdk sdk;

    public GollekLocalClientAdapter() {
    }

    public GollekLocalClientAdapter(GollekSdk sdk) {
        this.sdk = sdk;
    }

    @Override
    public InferenceResponse createCompletion(InferenceRequest request) {
        try {
            return sdk.createCompletion(request);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Local completion request failed", e);
        }
    }

    @Override
    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return sdk.createCompletionAsync(request);
    }

    @Override
    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
        return sdk.streamCompletion(request);
    }

    @Override
    public String submitAsyncJob(InferenceRequest request) {
        try {
            return sdk.submitAsyncJob(request);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to submit async job", e);
        }
    }

    @Override
    public AsyncJobStatus getJobStatus(String jobId) {
        try {
            return sdk.getJobStatus(jobId);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to get async job status: " + jobId, e);
        }
    }

    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) {
        try {
            return sdk.waitForJob(jobId, maxWaitTime, pollInterval);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed while waiting for async job: " + jobId, e);
        }
    }

    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) {
        try {
            return sdk.batchInference(batchRequest);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Batch inference failed", e);
        }
    }

    @Override
    public List<ProviderInfo> listAvailableProviders() {
        try {
            return sdk.listAvailableProviders();
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to list providers", e);
        }
    }

    @Override
    public ProviderInfo getProviderInfo(String providerId) {
        try {
            return sdk.getProviderInfo(providerId);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to get provider info: " + providerId, e);
        }
    }

    @Override
    public void setPreferredProvider(String providerId) {
        try {
            sdk.setPreferredProvider(providerId);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to set preferred provider: " + providerId, e);
        }
    }

    @Override
    public Optional<String> getPreferredProvider() {
        return sdk.getPreferredProvider();
    }

    @Override
    public List<ModelInfo> listModels() {
        try {
            return sdk.listModels();
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to list models", e);
        }
    }

    @Override
    public List<ModelInfo> listModels(int offset, int limit) {
        try {
            return sdk.listModels(offset, limit);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to list models with pagination", e);
        }
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelId) {
        try {
            return sdk.getModelInfo(modelId);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to get model info: " + modelId, e);
        }
    }

    @Override
    public void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) {
        try {
            sdk.pullModel(modelSpec, progressCallback);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to pull model: " + modelSpec, e);
        }
    }

    @Override
    public void deleteModel(String modelId) {
        try {
            sdk.deleteModel(modelId);
        } catch (SdkException e) {
            throw new GollekLocalClientException("Failed to delete model: " + modelId, e);
        }
    }

    @Override
    public McpRegistryManager mcpRegistry() {
        if (sdk instanceof McpRegistryProvider provider) {
            return provider.mcpRegistry();
        }
        throw new GollekLocalClientException("MCP registry manager is not available in this SDK instance", null);
    }
}
