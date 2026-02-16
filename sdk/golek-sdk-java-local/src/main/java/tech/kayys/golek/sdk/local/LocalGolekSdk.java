package tech.kayys.golek.sdk.local;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.enterprise.inject.spi.CDI;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.ArtifactLocation;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.ProviderInfo;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.engine.inference.InferenceService;
import tech.kayys.golek.model.repo.local.LocalModelRepository;
import tech.kayys.golek.spi.model.Pageable;
import tech.kayys.golek.model.download.DownloadProgressListener;
import java.time.Instant;

import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.exception.NonRetryableException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.golek.sdk.core.model.ModelInfo;
import tech.kayys.golek.sdk.core.model.PullProgress;
import tech.kayys.golek.sdk.core.tenant.TenantResolver;
import tech.kayys.golek.sdk.core.validation.RequestValidator;
import tech.kayys.golek.model.repo.hf.HuggingFaceClient;
import tech.kayys.golek.inference.gguf.ModelConverterService;
import java.nio.file.Files;

import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.nio.file.Paths;

import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Local implementation of the Golek SDK that runs within the same JVM as the
 * inference engine.
 * This implementation directly calls the internal services without HTTP
 * overhead.
 */
@ApplicationScoped
@Slf4j
public class LocalGolekSdk implements GolekSdk {

    private static final String DEFAULT_TENANT_ID = "default";

    @Inject
    InferenceService inferenceService;

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    TenantResolver tenantResolver;

    @Inject
    LocalModelRepository modelRepository;

    @Inject
    Instance<HuggingFaceClient> hfClientInstance;

    @Inject
    Instance<ModelConverterService> converterServiceInstance;

    // Default to GGUF provider
    private String preferredProvider = "gguf";

    /**
     * Resolves the current tenant ID using the injected TenantResolver.
     */
    protected String resolveTenantId() {
        try {
            return tenantResolver.resolveApiKey();
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
                    .map(chunk -> new StreamChunk(
                            chunk.requestId(),
                            chunk.sequenceNumber(),
                            chunk.token(),
                            chunk.isComplete(),
                            Instant.now()));
        } catch (NonRetryableException e) {
            log.error("Validation failed for streaming request: {}", e.getMessage());
            return Multi.createFrom().failure(e);
        } catch (Exception e) {
            log.error("Failed to initiate streaming completion: {}", e.getMessage(), e);
            return Multi.createFrom()
                    .failure(new SdkException("SDK_ERR_STREAM", "Failed to initiate streaming completion", e));
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
                    .apiKey(status.apiKey())
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
            return Multi.createFrom().items(batchRequest.getRequests().stream())
                    .onItem().transformToUniAndConcatenate(request -> inferenceService.inferAsync(request)
                            .onFailure().recoverWithItem(failure -> InferenceResponse.builder()
                                    .requestId(request.getRequestId())
                                    .model(request.getModel())
                                    .content("Error: " + failure.getMessage())
                                    .build()))
                    .collect().asList()
                    .await().indefinitely();
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
                    .orElseThrow(
                            () -> new SdkException("SDK_ERR_PROVIDER_NOT_FOUND", "Provider not found: " + providerId));
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
                // Default to HuggingFace
                provider = "huggingface";
                model = modelSpec;
            }

            if ("ollama".equals(provider)) {
                pullFromOllama(model, progressCallback);
            } else if ("huggingface".equals(provider)) {
                pullFromHuggingFace(model, progressCallback);
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
        Optional<Object> maybeClient = resolveBeanByClassName("tech.kayys.golek.provider.ollama.OllamaClient");
        if (maybeClient.isEmpty()) {
            throw new SdkException("SDK_ERR_PROVIDER_UNAVAILABLE", "Ollama client not available");
        }

        try {
            Object ollamaClient = maybeClient.get();
            Class<?> requestClass = Class.forName("tech.kayys.golek.provider.ollama.OllamaPullRequest");
            Object request = requestClass.getDeclaredConstructor().newInstance();

            requestClass.getMethod("setName", String.class).invoke(request, model);
            requestClass.getMethod("setStream", boolean.class).invoke(request, true);

            Object pullResult = ollamaClient.getClass().getMethod("pullModel", requestClass).invoke(ollamaClient,
                    request);
            if (!(pullResult instanceof Multi<?> stream)) {
                throw new SdkException("SDK_ERR_PROVIDER_UNAVAILABLE", "Ollama pull stream is unavailable");
            }

            stream.subscribe().with(
                    progress -> {
                        if (progressCallback != null) {
                            PullProgress pp = PullProgress.of(
                                    stringGetter(progress, "getStatus").orElse("downloading"),
                                    stringGetter(progress, "getDigest").orElse(model),
                                    longGetter(progress, "getTotal").orElse(0L),
                                    longGetter(progress, "getCompleted").orElse(0L));
                            progressCallback.accept(pp);
                        }
                    },
                    error -> log.error("Error pulling model: {}", error.getMessage()),
                    () -> log.info("Model pull complete: {}", model));
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_PULL", "Failed to pull model from Ollama", e);
        }
    }

    private void pullFromHuggingFace(String modelId, Consumer<PullProgress> progressCallback) throws SdkException {
        if (!hfClientInstance.isResolvable()) {
            throw new SdkException("SDK_ERR_PROVIDER_UNAVAILABLE", "HuggingFace client not available");
        }
        if (isLikelyMetaLlama(modelId) && !hasHuggingFaceToken()) {
            throw new SdkException(
                    "SDK_ERR_MODEL_PULL",
                    "Model '" + modelId
                            + "' is likely gated. Set wayang.inference.repository.huggingface.token (or env WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN) and accept model access on Hugging Face.");
        }

        try {
            HuggingFaceClient client = hfClientInstance.get();
            List<String> files = client.listFiles(modelId);

            // Prefer GGUF when available (default runtime), otherwise select a compatible
            // runtime.
            List<String> ggufFiles = files.stream()
                    .filter(f -> f.endsWith(".gguf"))
                    .toList();
            List<String> torchScriptFiles = files.stream()
                    .filter(this::isTorchScriptCandidateFile)
                    .toList();
            List<String> djlCheckpointFiles = files.stream()
                    .filter(this::isDjlCheckpointFile)
                    .toList();

            DownloadProgressListener listener = (bytesRead, totalBytes,
                    progress) -> {
                if (progressCallback != null) {
                    PullProgress pp = PullProgress.of(
                            "downloading",
                            modelId,
                            totalBytes,
                            bytesRead);
                    progressCallback.accept(pp);
                }
            };

            if (!ggufFiles.isEmpty()) {
                // Scenario A: Found GGUF files, download one
                String targetFile = ggufFiles.stream()
                        .filter(f -> f.toLowerCase().contains("q4_k_m"))
                        .findFirst()
                        .orElse(ggufFiles.get(0));

                log.info("Found GGUF file: {}, downloading...", targetFile);

                // Standardize GGUF model storage path
                Path outputDir = Paths.get(System.getProperty("user.home"), ".golek", "models", "gguf");
                Files.createDirectories(outputDir);

                // Use modelId as filename (replacing / with _)
                Path outputPath = outputDir.resolve(modelId.replace("/", "_") + ".gguf");

                client.downloadFile(modelId, targetFile, outputPath, listener);

                // Print path immediately for user visibility
                System.out.println("\n✓ Model saved to: " + outputPath);
                log.info("Model downloaded to: {}", outputPath);

                // Register model in repository so it persists
                registerDownloadedModel(modelId, outputPath);

            } else if (!torchScriptFiles.isEmpty()) {
                String targetFile = pickBestTorchScriptFile(torchScriptFiles);
                log.info("Found LibTorch candidate file: {}, downloading...", targetFile);

                Path outputBase = Paths.get(System.getProperty("user.home"), ".golek", "models", "torchscript");
                Path outputPath = outputBase.resolve(modelId + fileExtension(targetFile));
                if (outputPath.getParent() != null) {
                    Files.createDirectories(outputPath.getParent());
                }

                client.downloadFile(modelId, targetFile, outputPath, listener);

                System.out.println("\n✓ Model saved to: " + outputPath);
                log.info("Model downloaded to: {}", outputPath);

                registerDownloadedModel(modelId, outputPath, inferFormatFromFileName(targetFile));

            } else if (!djlCheckpointFiles.isEmpty()) {
                // Raw HF checkpoints (.bin/.safetensors/.pth) are not directly executable
                // in our Java-only runtime path. Try GGUF fallback or conversion instead.
                log.info("Found checkpoint files for {} but no GGUF/TorchScript artifacts", modelId);
                if (preferPrebuiltGGUFFallback()) {
                    boolean usedFallback = tryLikelyGGUFFallback(modelId, progressCallback);
                    if (usedFallback) {
                        return;
                    }
                }
                if (!isGgufAutoConversionEnabled()) {
                    throw new SdkException(
                            "SDK_ERR_UNSUPPORTED_FORMAT",
                            "Repository has checkpoint files only and auto-conversion is disabled. "
                                    + "Enable conversion or use a GGUF/TorchScript model.");
                }
                if (!converterServiceInstance.isResolvable() || !converterServiceInstance.get().isAvailable()) {
                    if (!modelId.toUpperCase().endsWith("-GGUF")) {
                        attemptGGUFFallback(modelId, progressCallback);
                        return;
                    }
                    throw new SdkException(
                            "SDK_ERR_UNSUPPORTED_FORMAT",
                            "Repository has PyTorch checkpoints only (.bin/.safetensors/.pth). "
                                    + "Use a GGUF or TorchScript model for Java runtime.");
                }

                log.info("No GGUF/TorchScript file found for {}, downloading for conversion...", modelId);

                Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "golek-conversion",
                        modelId.replace("/", "_"));
                client.downloadRepository(modelId, tempDir, listener);

                Path outputDir = Paths.get(System.getProperty("user.home"), ".golek", "models", "gguf");
                Files.createDirectories(outputDir);
                Path outputFile = outputDir.resolve(modelId.replace("/", "_") + ".gguf");

                log.info("Converting model to {}", outputFile);
                converterServiceInstance.get().convert(tempDir, outputFile, configuredConverterOutType());

                System.out.println("\n✓ Model saved to: " + outputFile);
                log.info("Model converted and saved to: {}", outputFile);

                registerDownloadedModel(modelId, outputFile);
            } else {
                // Scenario B: No GGUF, download and convert
                if (!isGgufAutoConversionEnabled()) {
                    throw new SdkException(
                            "SDK_ERR_UNSUPPORTED_FORMAT",
                            "No GGUF artifact found and auto-conversion is disabled for this model.");
                }
                if (!converterServiceInstance.isResolvable() || !converterServiceInstance.get().isAvailable()) {
                    // Try fallback if not already a GGUF variant
                    if (!modelId.toUpperCase().endsWith("-GGUF")) {
                        attemptGGUFFallback(modelId, progressCallback);
                        return;
                    }
                    throw new SdkException("SDK_ERR_CONVERSION",
                            "Model conversion service not available and no GGUF found for: " + modelId);
                }

                log.info("No GGUF file found for {}, downloading for conversion...", modelId);

                Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "golek-conversion",
                        modelId.replace("/", "_"));
                client.downloadRepository(modelId, tempDir, listener);

                Path outputDir = Paths.get(System.getProperty("user.home"), ".golek", "models", "gguf");
                Files.createDirectories(outputDir);
                Path outputFile = outputDir.resolve(modelId.replace("/", "_") + ".gguf");

                log.info("Converting model to {}", outputFile);
                converterServiceInstance.get().convert(tempDir, outputFile, configuredConverterOutType());

                System.out.println("\n✓ Model saved to: " + outputFile);
                log.info("Model converted and saved to: {}", outputFile);

                // Register converted model
                registerDownloadedModel(modelId, outputFile);
            }

        } catch (Exception e) {
            String reason = rootCauseMessage(e);
            String message = "Failed to pull/convert model from HuggingFace";
            if (reason != null && !reason.isBlank()) {
                message = message + ": " + reason;
            }
            if (isLikelyGatedHuggingFaceModel(reason)) {
                if (hasHuggingFaceToken()) {
                    message = message
                            + " (Hugging Face token detected, but access is denied. "
                            + "Accept access for this model with the same HF account as the token, "
                            + "and ensure token scope includes Read permission.)";
                } else {
                    message = message
                            + " (this repository is likely gated/private; set a valid HF token and ensure access is accepted)";
                }
            }
            throw new SdkException("SDK_ERR_MODEL_PULL", message, e);
        }
    }

    private void attemptGGUFFallback(String originalModelId, Consumer<PullProgress> progressCallback)
            throws SdkException {
        String fallbackId = originalModelId + "-GGUF";
        log.info("Checking for GGUF variant: {}", fallbackId);
        try {
            // Check if fallback model exists and has GGUF files
            HuggingFaceClient client = hfClientInstance.get();
            try {
                // We use listFiles to check existence and content in one go
                List<String> files = client.listFiles(fallbackId);
                boolean hasGGUF = files.stream().anyMatch(f -> f.endsWith(".gguf"));

                if (hasGGUF) {
                    log.info("Found GGUF variant: {}. Switching download...", fallbackId);
                    pullFromHuggingFace(fallbackId, progressCallback);
                    return;
                }
            } catch (Exception ignored) {
                // Fallback model doesn't exist or other error, ignore and throw original
                // exception
            }
        } catch (Exception e) {
            log.warn("Failed during GGUF fallback check", e);
        }
        // If we reach here, fallback failed
        throw new SdkException("SDK_ERR_CONVERSION",
                "Model conversion service not available and no GGUF found for: " + originalModelId +
                        ". Also failed to find GGUF files in likely fallback: " + fallbackId);
    }

    private ProviderInfo toProviderInfo(LLMProvider provider) {
        var health = provider.health()
                .await()
                .atMost(Duration.ofSeconds(5));
        var healthStatus = health.status();
        var metadata = provider.metadata();
        String description = metadata != null ? metadata.getDescription() : null;
        String vendor = metadata != null ? metadata.getVendor() : null;
        var capabilities = provider.capabilities();
        java.util.Map<String, Object> providerMetadata = new java.util.HashMap<>();
        if (health.message() != null && !health.message().isBlank()) {
            providerMetadata.put("healthMessage", health.message());
        }
        if (health.details() != null && !health.details().isEmpty()) {
            providerMetadata.put("healthDetails", health.details());
        }

        return ProviderInfo.builder()
                .id(provider.id())
                .name(provider.name())
                .version(provider.version())
                .description(description)
                .vendor(vendor)
                .healthStatus(healthStatus)
                .capabilities(capabilities)
                .supportedModels(java.util.Set.of())
                .metadata(providerMetadata)
                .build();
    }

    private ModelInfo toModelInfo(ModelManifest manifest) {
        Long sizeBytes = null;
        String format = null;

        java.util.Map<String, Object> metadata = manifest.metadata() != null
                ? new java.util.HashMap<>(manifest.metadata())
                : new java.util.HashMap<>();

        if (manifest.artifacts() != null && !manifest.artifacts().isEmpty()) {
            sizeBytes = manifest.artifacts().values().stream()
                    .filter(java.util.Objects::nonNull)
                    .mapToLong(a -> a.size() != null ? a.size() : 0)
                    .sum();
            format = manifest.artifacts().keySet().stream()
                    .filter(java.util.Objects::nonNull)
                    .map(Enum::name)
                    .findFirst()
                    .orElse(null);

            // Add path to metadata for CLI to display
            manifest.artifacts().values().stream()
                    .filter(java.util.Objects::nonNull)
                    .findFirst()
                    .ifPresent(a -> metadata.put("path", a.uri()));
        }

        return ModelInfo.builder()
                .modelId(manifest.modelId())
                .name(manifest.name())
                .version(manifest.version())
                .apiKey(manifest.apiKey())
                .format(format)
                .sizeBytes(sizeBytes)
                .createdAt(manifest.createdAt())
                .updatedAt(manifest.updatedAt())
                .metadata(metadata)
                .build();
    }

    /**
     * Register a downloaded model in the repository so it persists across sessions
     */
    private void registerDownloadedModel(String modelId, Path filePath) {
        registerDownloadedModel(modelId, filePath, tech.kayys.golek.spi.model.ModelFormat.GGUF);
    }

    private void registerDownloadedModel(String modelId, Path filePath, tech.kayys.golek.spi.model.ModelFormat format) {
        try {
            long fileSize = Files.isDirectory(filePath) ? directorySize(filePath) : Files.size(filePath);
            String mimeType = Files.isDirectory(filePath) ? "inode/directory" : "application/octet-stream";

            // Create artifact location
            ArtifactLocation artifactLocation = new ArtifactLocation(
                    filePath.toUri().toString(),
                    null, // checksum - optional for now
                    fileSize,
                    mimeType);

            // Create manifest with artifact
            ModelManifest manifest = ModelManifest.builder()
                    .modelId(modelId)
                    .name(modelId)
                    .version("1.0.0")
                    .requestId("local-" + modelId.replace("/", "_"))
                    .path(filePath.toUri().toString())
                    .apiKey(DEFAULT_TENANT_ID)
                    .artifacts(java.util.Map.of(format, artifactLocation))
                    .metadata(java.util.Map.of("source", "huggingface", "path", filePath.toString()))
                    .build();

            // Save to repository
            modelRepository.save(manifest).await().indefinitely();
            log.info("Registered model {} in repository", modelId);

        } catch (Exception e) {
            log.warn("Failed to register model {} in repository: {}", modelId, e.getMessage());
            // Don't fail the download if registration fails
        }
    }

    private boolean isTorchScriptCandidateFile(String fileName) {
        String f = fileName.toLowerCase();
        return f.endsWith(".pt") || f.endsWith(".pts") || f.endsWith(".jit") || f.endsWith(".ts");
    }

    private boolean isDjlCheckpointFile(String fileName) {
        String f = fileName.toLowerCase();
        return f.endsWith(".bin") || f.endsWith(".safetensors") || f.endsWith(".pth");
    }

    private String pickBestTorchScriptFile(List<String> candidates) {
        return candidates.stream()
                .filter(f -> f.endsWith(".pt") || f.endsWith(".jit") || f.endsWith(".ts") || f.endsWith(".pts"))
                .findFirst()
                .orElse(candidates.get(0));
    }

    private String pickBestDjlCheckpointFile(List<String> candidates) {
        return candidates.stream()
                .filter(f -> f.endsWith(".safetensors"))
                .findFirst()
                .orElseGet(() -> candidates.stream()
                        .filter(f -> f.endsWith(".bin"))
                        .findFirst()
                        .orElse(candidates.get(0)));
    }

    private String fileExtension(String fileName) {
        int idx = fileName.lastIndexOf('.');
        return idx >= 0 ? fileName.substring(idx) : "";
    }

    private tech.kayys.golek.spi.model.ModelFormat inferFormatFromFileName(String fileName) {
        String f = fileName.toLowerCase();
        if (f.endsWith(".pt") || f.endsWith(".pts") || f.endsWith(".jit") || f.endsWith(".ts")) {
            return tech.kayys.golek.spi.model.ModelFormat.TORCHSCRIPT;
        }
        if (f.endsWith(".safetensors")) {
            return tech.kayys.golek.spi.model.ModelFormat.SAFETENSORS;
        }
        if (f.endsWith(".pth") || f.endsWith(".bin")) {
            return tech.kayys.golek.spi.model.ModelFormat.PYTORCH;
        }
        return tech.kayys.golek.spi.model.ModelFormat.UNKNOWN;
    }

    private long directorySize(Path directory) {
        try (var stream = Files.walk(directory)) {
            return stream
                    .filter(Files::isRegularFile)
                    .mapToLong(p -> {
                        try {
                            return Files.size(p);
                        } catch (Exception ignored) {
                            return 0L;
                        }
                    })
                    .sum();
        } catch (Exception e) {
            return 0L;
        }
    }

    private Optional<Object> resolveBeanByClassName(String className) {
        try {
            Class<?> clazz = Class.forName(className);
            var instance = CDI.current().select(clazz);
            if (instance.isResolvable()) {
                return Optional.ofNullable(instance.get());
            }
        } catch (ClassNotFoundException ignored) {
            // Optional extension is not on classpath.
        } catch (Exception e) {
            log.warn("Failed to resolve optional bean {}", className, e);
        }
        return Optional.empty();
    }

    private Optional<String> stringGetter(Object target, String getterName) {
        try {
            Object value = target.getClass().getMethod(getterName).invoke(target);
            return Optional.ofNullable(value != null ? String.valueOf(value) : null);
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Long> longGetter(Object target, String getterName) {
        try {
            Object value = target.getClass().getMethod(getterName).invoke(target);
            if (value instanceof Number n) {
                return Optional.of(n.longValue());
            }
        } catch (Exception ignored) {
            return Optional.empty();
        }
        return Optional.empty();
    }

    private String rootCauseMessage(Throwable throwable) {
        Throwable current = throwable;
        String message = null;
        String type = null;
        int guard = 0;
        while (current != null && guard++ < 16) {
            type = current.getClass().getSimpleName();
            if (current.getMessage() != null && !current.getMessage().isBlank()) {
                message = current.getMessage();
            }
            current = current.getCause();
        }
        if (message != null && !message.isBlank()) {
            return message;
        }
        return type != null ? type : null;
    }

    private boolean isLikelyGatedHuggingFaceModel(String message) {
        if (message == null) {
            return false;
        }
        String lower = message.toLowerCase();
        return lower.contains("401")
                || lower.contains("403")
                || lower.contains("unauthorized")
                || lower.contains("forbidden")
                || lower.contains("gated")
                || lower.contains("access to model")
                || lower.contains("access denied");
    }

    private boolean isLikelyMetaLlama(String modelId) {
        return modelId != null && modelId.toLowerCase().startsWith("meta-llama/");
    }

    private boolean hasHuggingFaceToken() {
        String propertyToken = System.getProperty("wayang.inference.repository.huggingface.token");
        if (propertyToken != null && !propertyToken.isBlank()) {
            return true;
        }

        String[] keys = {
                "WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN"
        };
        for (String key : keys) {
            String value = System.getenv(key);
            if (value != null && !value.isBlank()) {
                return true;
            }
        }
        return false;
    }

    private boolean isGgufAutoConversionEnabled() {
        String value = System.getProperty("golek.gguf.converter.auto");
        if (value == null || value.isBlank()) {
            return true;
        }
        return Boolean.parseBoolean(value);
    }

    private String configuredConverterOutType() {
        String outType = System.getProperty("golek.gguf.converter.outtype");
        if (outType == null || outType.isBlank()) {
            return null;
        }
        return outType.trim();
    }

    private boolean preferPrebuiltGGUFFallback() {
        String value = System.getProperty("golek.gguf.prefer-prebuilt", "true");
        return Boolean.parseBoolean(value);
    }

    private boolean tryLikelyGGUFFallback(String modelId, Consumer<PullProgress> progressCallback) {
        String fallbackId = modelId + "-GGUF";
        log.info("Checking likely prebuilt GGUF fallback: {}", fallbackId);
        if (!hfClientInstance.isResolvable()) {
            return false;
        }
        try {
            HuggingFaceClient client = hfClientInstance.get();
            List<String> files = client.listFiles(fallbackId);
            boolean hasGGUF = files.stream().anyMatch(f -> f.endsWith(".gguf"));
            if (!hasGGUF) {
                return false;
            }
            log.info("Found GGUF fallback repository: {}", fallbackId);
            pullFromHuggingFace(fallbackId, progressCallback);
            return true;
        } catch (Exception e) {
            log.debug("GGUF fallback check failed for {}: {}", fallbackId, e.getMessage());
            return false;
        }
    }
}
