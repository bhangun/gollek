package tech.kayys.gollek.sdk.local;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.engine.inference.InferenceService;
import tech.kayys.gollek.engine.model.CachedModelRepository;
import tech.kayys.gollek.engine.service.AsyncJobManager;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.exception.SdkException;
import tech.kayys.gollek.sdk.core.model.AsyncJobStatus;
import tech.kayys.gollek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.gollek.sdk.core.model.ModelInfo;
import tech.kayys.gollek.sdk.core.model.PullProgress;
import tech.kayys.gollek.sdk.core.mcp.McpRegistryManager;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.Pageable;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

@ApplicationScoped
public class LocalGollekSdk implements GollekSdk, McpRegistryProvider {

    private static final int MAX_MODEL_SCAN_SIZE = 10_000;

    @Inject
    InferenceService inferenceService;

    @Inject
    AsyncJobManager asyncJobManager;

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    CachedModelRepository modelRepository;

    @Inject
    McpRegistryManager mcpRegistryManager;

    private volatile String preferredProvider;

    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            return inferenceService.inferAsync(enrichRequest(request))
                    .await()
                    .indefinitely();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_INFERENCE", "Local inference failed", e);
        }
    }

    @Override
    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return createCompletion(request);
            } catch (SdkException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
        try {
            return inferenceService.inferStream(enrichRequest(request))
                    .onItem().transform(chunk -> new StreamChunk(
                            chunk.requestId(),
                            chunk.sequenceNumber(),
                            chunk.token(),
                            chunk.isComplete(),
                            Instant.now()));
        } catch (Exception e) {
            return Multi.createFrom().failure(new SdkException("SDK_ERR_STREAM", "Local streaming failed", e));
        }
    }

    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        try {
            return inferenceService.submitAsyncJob(enrichRequest(request))
                    .await()
                    .indefinitely();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_ASYNC_SUBMIT", "Failed to submit local async job", e);
        }
    }

    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        try {
            tech.kayys.gollek.spi.inference.AsyncJobStatus status = asyncJobManager.getStatus(jobId);
            if (status == null) {
                throw new SdkException("SDK_ERR_JOB_NOT_FOUND", "Async job not found: " + jobId);
            }
            return mapAsyncStatus(status);
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_ASYNC_STATUS", "Failed to fetch async job status", e);
        }
    }

    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        long start = System.currentTimeMillis();
        long maxMillis = maxWaitTime.toMillis();

        while (System.currentTimeMillis() - start < maxMillis) {
            AsyncJobStatus status = getJobStatus(jobId);
            if (status.isComplete()) {
                return status;
            }
            try {
                Thread.sleep(pollInterval.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new SdkException("SDK_ERR_INTERRUPTED", "Interrupted while waiting for async job", e);
            }
        }

        throw new SdkException("SDK_ERR_TIMEOUT", "Async job timed out: " + jobId);
    }

    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        if (batchRequest == null || batchRequest.getRequests() == null) {
            return List.of();
        }
        List<InferenceResponse> responses = new ArrayList<>();
        for (InferenceRequest request : batchRequest.getRequests()) {
            responses.add(createCompletion(request));
        }
        return responses;
    }

    @Override
    public List<ProviderInfo> listAvailableProviders() throws SdkException {
        try {
            return providerRegistry.getAllProviders().stream()
                    .map(this::toProviderInfo)
                    .sorted(Comparator.comparing(ProviderInfo::id))
                    .toList();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_PROVIDER_LIST", "Failed to list providers", e);
        }
    }

    @Override
    public ProviderInfo getProviderInfo(String providerId) throws SdkException {
        return providerRegistry.getProvider(providerId)
                .map(this::toProviderInfo)
                .orElseThrow(() -> new SdkException("SDK_ERR_PROVIDER_NOT_FOUND",
                        "Provider not found: " + providerId));
    }

    @Override
    public void setPreferredProvider(String providerId) throws SdkException {
        if (!providerRegistry.hasProvider(providerId)) {
            throw new SdkException("SDK_ERR_PROVIDER_NOT_FOUND", "Provider not found: " + providerId);
        }
        this.preferredProvider = providerId;
    }

    @Override
    public Optional<String> getPreferredProvider() {
        return Optional.ofNullable(preferredProvider);
    }

    @Override
    public List<ModelInfo> listModels() throws SdkException {
        return listModels(0, MAX_MODEL_SCAN_SIZE);
    }

    @Override
    public List<ModelInfo> listModels(int offset, int limit) throws SdkException {
        try {
            if (limit <= 0) {
                return List.of();
            }
            List<ModelManifest> models = modelRepository.list(resolveApiKey(), Pageable.of(0, MAX_MODEL_SCAN_SIZE))
                    .await().indefinitely();
            if (models == null || models.isEmpty()) {
                return List.of();
            }
            int start = Math.max(offset, 0);
            if (start >= models.size()) {
                return List.of();
            }
            int end = Math.min(start + limit, models.size());
            return models.subList(start, end).stream()
                    .map(this::toModelInfo)
                    .toList();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_LIST", "Failed to list models", e);
        }
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelId) throws SdkException {
        try {
            ModelManifest manifest = modelRepository.findById(modelId, resolveApiKey())
                    .await()
                    .indefinitely();
            return Optional.ofNullable(manifest).map(this::toModelInfo);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_INFO", "Failed to get model info for: " + modelId, e);
        }
    }

    @Override
    public void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException {
        throw new SdkException("SDK_ERR_UNSUPPORTED",
                "Local SDK pullModel is not implemented. Use runtime/API model management.");
    }

    @Override
    public void deleteModel(String modelId) throws SdkException {
        try {
            modelRepository.delete(modelId, resolveApiKey())
                    .await()
                    .indefinitely();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_DELETE", "Failed to delete model: " + modelId, e);
        }
    }

    @Override
    public McpRegistryManager mcpRegistry() {
        return mcpRegistryManager;
    }

    private InferenceRequest enrichRequest(InferenceRequest request) {
        if (preferredProvider == null || request.getPreferredProvider().isPresent()) {
            return request;
        }
        return request.toBuilder()
                .preferredProvider(preferredProvider)
                .build();
    }

    private String resolveApiKey() {
        return ApiKeyConstants.COMMUNITY_API_KEY;
    }

    private AsyncJobStatus mapAsyncStatus(tech.kayys.gollek.spi.inference.AsyncJobStatus status) {
        return AsyncJobStatus.builder()
                .jobId(status.jobId())
                .apiKey(status.apiKey())
                .status(status.status())
                .result(status.result())
                .error(status.error())
                .submittedAt(status.submittedAt())
                .completedAt(status.completedAt())
                .build();
    }

    private ProviderInfo toProviderInfo(LLMProvider provider) {
        ProviderHealth.Status status;
        try {
            status = provider.health().await().indefinitely().status();
        } catch (Exception e) {
            status = ProviderHealth.Status.UNKNOWN;
        }

        Map<String, Object> metadata = new HashMap<>();
        if (provider.metadata() != null) {
            if (provider.metadata().getProviderId() != null) {
                metadata.put("providerId", provider.metadata().getProviderId());
            }
            if (provider.metadata().getHomepage() != null) {
                metadata.put("homepage", provider.metadata().getHomepage());
            }
        }

        return ProviderInfo.builder()
                .id(provider.id())
                .name(provider.name())
                .version(provider.version())
                .description(provider.metadata() != null ? provider.metadata().getDescription() : null)
                .vendor(provider.metadata() != null ? provider.metadata().getVendor() : null)
                .healthStatus(status)
                .capabilities(provider.capabilities())
                .supportedModels(Set.of())
                .metadata(metadata)
                .build();
    }

    private ModelInfo toModelInfo(ModelManifest model) {
        String format = model.artifacts() != null && !model.artifacts().isEmpty()
                ? model.artifacts().keySet().stream().findFirst().map(ModelFormat::getId).orElse(null)
                : null;

        Long sizeBytes = null;
        if (model.metadata() != null) {
            Object fromMeta = model.metadata().get("sizeBytes");
            if (fromMeta instanceof Number n) {
                sizeBytes = n.longValue();
            }
        }

        return ModelInfo.builder()
                .modelId(model.modelId())
                .name(model.name())
                .version(model.version())
                .apiKey(model.apiKey())
                .format(format)
                .sizeBytes(sizeBytes)
                .quantization(model.metadata() != null && model.metadata().get("quantization") != null
                        ? model.metadata().get("quantization").toString()
                        : null)
                .createdAt(model.createdAt())
                .updatedAt(model.updatedAt())
                .metadata(model.metadata() != null ? model.metadata() : Map.of())
                .build();
    }
}
