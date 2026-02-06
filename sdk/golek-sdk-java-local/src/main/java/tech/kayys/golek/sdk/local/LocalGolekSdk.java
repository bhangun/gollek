package tech.kayys.golek.sdk.local;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.ProviderInfo;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.engine.inference.InferenceService;
import tech.kayys.golek.model.core.LocalModelRepository;
import tech.kayys.golek.model.core.Pageable;
import tech.kayys.golek.provider.ollama.OllamaClient;
import tech.kayys.golek.provider.ollama.OllamaPullRequest;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.exception.NonRetryableException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.golek.sdk.core.model.ModelInfo;
import tech.kayys.golek.sdk.core.model.PullProgress;
import tech.kayys.golek.sdk.core.tenant.TenantResolver;
import tech.kayys.golek.sdk.core.validation.RequestValidator;
import lombok.extern.slf4j.Slf4j;

import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Local implementation of the Golek SDK that runs within the same JVM as the inference engine.
 * This implementation directly calls the internal services without HTTP overhead.
 */
@ApplicationScoped
@Slf4j
public class LocalGolekSdk implements GolekSdk {
    
    @Inject
    InferenceService inferenceService;
    
    @Inject
    ProviderRegistry providerRegistry;
    
    @Inject
    TenantResolver tenantResolver;
    
    @Inject
    LocalModelRepository modelRepository;
    
    @Inject
    Instance<OllamaClient> ollamaClientInstance;
    
    private String preferredProvider;
    
    /**
     * Resolves the current tenant ID using the injected TenantResolver.
     */
    protected String resolveTenantId() {
        try {
            return tenantResolver.resolveTenantId();
        } catch (Exception e) {
            log.warn("Failed to resolve tenant ID, using 'default': {}", e.getMessage());
            return "default";
        }
    }

    // ==================== Inference Operations ====================

    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            RequestValidator.validate(request);
            log.debug("Creating completion: requestId={}, model={}", 
                request.getRequestId(), request.getModel());
            return inferenceService.inferAsync(request).await().indefinitely();
        } catch (NonRetryableException e) {
            log.error("Validation failed for completion request: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            log.error("Failed to create completion: {}", e.getMessage(), e);
            throw new SdkException("SDK_ERR_INFERENCE", "Failed to create completion", e);
        }
    }
    
    @Override
    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        log.debug("Creating completion async: requestId={}", request.getRequestId());
        return inferenceService.inferAsync(request).subscribeAsCompletionStage();
    }
    
    @Override
    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
        try {
            RequestValidator.validate(request);
            log.debug("Streaming completion: requestId={}, model={}", 
                request.getRequestId(), request.getModel());
            return inferenceService.inferStream(request)
                .map(chunk -> new StreamChunk(chunk.index(), chunk.delta(), chunk.isFinal()));
        } catch (NonRetryableException e) {
            log.error("Validation failed for streaming request: {}", e.getMessage());
            return Multi.createFrom().failure(e);
        } catch (Exception e) {
            log.error("Failed to initiate streaming completion: {}", e.getMessage(), e);
            return Multi.createFrom().failure(new SdkException("SDK_ERR_STREAM", "Failed to initiate streaming completion", e));
        }
    }

    // ==================== Job Operations ====================

    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        try {
            log.debug("Submitting async job: requestId={}", request.getRequestId());
            return inferenceService.submitAsyncJob(request).await().indefinitely();
        } catch (Exception e) {
            log.error("Failed to submit async job: {}", e.getMessage(), e);
            throw new SdkException("SDK_ERR_JOB_SUBMIT", "Failed to submit async job", e);
        }
    }
    
    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        try {
            log.debug("Getting job status: jobId={}", jobId);
            var status = inferenceService.getJobStatus(jobId, resolveTenantId()).await().indefinitely();
            return AsyncJobStatus.builder()
                .jobId(status.jobId())
                .requestId(status.requestId())
                .tenantId(status.tenantId())
                .status(status.status())
                .result(status.result())
                .error(status.error())
                .submittedAt(status.submittedAt())
                .completedAt(status.completedAt())
                .build();
        } catch (Exception e) {
            log.error("Failed to get job status for {}: {}", jobId, e.getMessage(), e);
            throw new SdkException("SDK_ERR_JOB_STATUS", "Failed to get job status", e);
        }
    }
    
    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        log.debug("Waiting for job: jobId={}, maxWaitTime={}, pollInterval={}", jobId, maxWaitTime, pollInterval);
        long startTime = System.currentTimeMillis();
        long maxWaitMillis = maxWaitTime.toMillis();
        
        while (System.currentTimeMillis() - startTime < maxWaitMillis) {
            AsyncJobStatus status = getJobStatus(jobId);
            if (status.isComplete()) {
                return status;
            }
            try {
                Thread.sleep(pollInterval.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new SdkException("SDK_ERR_INTERRUPTED", "Job polling interrupted", e);
            }
        }
        throw new SdkException("SDK_ERR_TIMEOUT", "Job " + jobId + " did not complete within the specified time");
    }
    
    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        try {
            log.debug("Starting batch inference: size={}", batchRequest.getRequests().size());
            var internalBatchRequest = new InferenceService.BatchInferenceRequest(
                batchRequest.getRequests(),
                batchRequest.getMaxConcurrent() != null ? batchRequest.getMaxConcurrent() : 5
            );
            return inferenceService.batchInfer(internalBatchRequest, resolveTenantId()).await().indefinitely();
        } catch (Exception e) {
            log.error("Failed to perform batch inference: {}", e.getMessage(), e);
            throw new SdkException("SDK_ERR_BATCH", "Failed to perform batch inference", e);
        }
    }

    // ==================== Provider Operations ====================

    @Override
    public List<ProviderInfo> listAvailableProviders() throws SdkException {
        try {
            log.debug("Listing available providers");
            return providerRegistry.getAllProviders().stream()
                .map(this::toProviderInfo)
                .toList();
        } catch (Exception e) {
            log.error("Failed to list providers: {}", e.getMessage(), e);
            throw new SdkException("SDK_ERR_PROVIDER_LIST", "Failed to list providers", e);
        }
    }
    
    @Override
    public ProviderInfo getProviderInfo(String providerId) throws SdkException {
        try {
            log.debug("Getting provider info: providerId={}", providerId);
            LLMProvider provider = providerRegistry.getProvider(providerId)
                .orElseThrow(() -> new SdkException("SDK_ERR_PROVIDER_NOT_FOUND", "Provider not found: " + providerId));
            return toProviderInfo(provider);
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            log.error("Failed to get provider info for {}: {}", providerId, e.getMessage(), e);
            throw new SdkException("SDK_ERR_PROVIDER_INFO", "Failed to get provider info", e);
        }
    }
    
    @Override
    public void setPreferredProvider(String providerId) throws SdkException {
        try {
            log.debug("Setting preferred provider: providerId={}", providerId);
            if (!providerRegistry.hasProvider(providerId)) {
                throw new SdkException("SDK_ERR_PROVIDER_NOT_FOUND", "Provider not available: " + providerId);
            }
            this.preferredProvider = providerId;
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            log.error("Failed to set preferred provider {}: {}", providerId, e.getMessage(), e);
            throw new SdkException("SDK_ERR_PROVIDER_SET", "Failed to set preferred provider", e);
        }
    }
    
    @Override
    public Optional<String> getPreferredProvider() {
        return Optional.ofNullable(preferredProvider);
    }

    // ==================== Model Operations ====================

    @Override
    public List<ModelInfo> listModels() throws SdkException {
        return listModels(0, 100);
    }

    @Override
    public List<ModelInfo> listModels(int offset, int limit) throws SdkException {
        try {
            log.debug("Listing models: offset={}, limit={}", offset, limit);
            String tenantId = resolveTenantId();
            List<ModelManifest> manifests = modelRepository.list(tenantId, new Pageable(offset, limit))
                .await().indefinitely();
            return manifests.stream()
                .map(this::toModelInfo)
                .toList();
        } catch (Exception e) {
            log.error("Failed to list models: {}", e.getMessage(), e);
            throw new SdkException("SDK_ERR_MODEL_LIST", "Failed to list models", e);
        }
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelId) throws SdkException {
        try {
            log.debug("Getting model info: modelId={}", modelId);
            String tenantId = resolveTenantId();
            ModelManifest manifest = modelRepository.findById(modelId, tenantId)
                .await().indefinitely();
            return Optional.ofNullable(manifest).map(this::toModelInfo);
        } catch (Exception e) {
            log.error("Failed to get model info for {}: {}", modelId, e.getMessage(), e);
            throw new SdkException("SDK_ERR_MODEL_INFO", "Failed to get model info", e);
        }
    }

    @Override
    public void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException {
        try {
            log.info("Pulling model: {}", modelSpec);
            
            // Parse model spec
            String provider;
            String model;
            if (modelSpec.startsWith("hf:")) {
                provider = "huggingface";
                model = modelSpec.substring(3);
            } else if (modelSpec.startsWith("ollama:")) {
                provider = "ollama";
                model = modelSpec.substring(7);
            } else {
                // Default to Ollama
                provider = "ollama";
                model = modelSpec;
            }

            if ("ollama".equals(provider)) {
                pullFromOllama(model, progressCallback);
            } else if ("huggingface".equals(provider)) {
                throw new SdkException("SDK_ERR_NOT_IMPLEMENTED", "HuggingFace pull not yet implemented");
            } else {
                throw new SdkException("SDK_ERR_UNKNOWN_PROVIDER", "Unknown model provider: " + provider);
            }
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            log.error("Failed to pull model {}: {}", modelSpec, e.getMessage(), e);
            throw new SdkException("SDK_ERR_MODEL_PULL", "Failed to pull model", e);
        }
    }

    @Override
    public void deleteModel(String modelId) throws SdkException {
        try {
            log.info("Deleting model: {}", modelId);
            String tenantId = resolveTenantId();
            modelRepository.delete(modelId, tenantId).await().indefinitely();
        } catch (Exception e) {
            log.error("Failed to delete model {}: {}", modelId, e.getMessage(), e);
            throw new SdkException("SDK_ERR_MODEL_DELETE", "Failed to delete model", e);
        }
    }

    // ==================== Private Helpers ====================

    private void pullFromOllama(String model, Consumer<PullProgress> progressCallback) throws SdkException {
        if (!ollamaClientInstance.isResolvable()) {
            throw new SdkException("SDK_ERR_PROVIDER_UNAVAILABLE", "Ollama client not available");
        }

        OllamaClient ollamaClient = ollamaClientInstance.get();
        OllamaPullRequest request = new OllamaPullRequest();
        request.setName(model);
        request.setStream(true);

        ollamaClient.pullModel(request)
            .subscribe().with(
                progress -> {
                    if (progressCallback != null) {
                        PullProgress pp = PullProgress.of(
                            progress.getStatus(),
                            progress.getDigest(),
                            progress.getTotal() != null ? progress.getTotal() : 0,
                            progress.getCompleted() != null ? progress.getCompleted() : 0
                        );
                        progressCallback.accept(pp);
                    }
                },
                error -> log.error("Error pulling model: {}", error.getMessage()),
                () -> log.info("Model pull complete: {}", model)
            );
    }
    
    private ProviderInfo toProviderInfo(LLMProvider provider) {
        var healthStatus = provider.health()
            .await()
            .atMost(Duration.ofSeconds(5))
            .status();
            
        return ProviderInfo.builder()
            .id(provider.id())
            .name(provider.name())
            .version(provider.version())
            .description(provider.metadata().description())
            .vendor(provider.metadata().vendor())
            .healthStatus(healthStatus)
            .capabilities(provider.capabilities())
            .supportedModels(java.util.Set.of())
            .metadata(java.util.Map.of())
            .build();
    }

    private ModelInfo toModelInfo(ModelManifest manifest) {
        Long sizeBytes = null;
        String format = null;
        
        if (manifest.artifacts() != null && !manifest.artifacts().isEmpty()) {
            sizeBytes = manifest.artifacts().values().stream()
                .mapToLong(a -> a.sizeBytes() != null ? a.sizeBytes() : 0)
                .sum();
            format = manifest.artifacts().values().stream()
                .map(a -> a.format())
                .filter(f -> f != null)
                .findFirst()
                .orElse(null);
        }

        return ModelInfo.builder()
            .modelId(manifest.modelId())
            .name(manifest.name())
            .version(manifest.version())
            .tenantId(manifest.tenantId())
            .format(format)
            .sizeBytes(sizeBytes)
            .createdAt(manifest.createdAt())
            .updatedAt(manifest.updatedAt())
            .metadata(manifest.metadata())
            .build();
    }
}