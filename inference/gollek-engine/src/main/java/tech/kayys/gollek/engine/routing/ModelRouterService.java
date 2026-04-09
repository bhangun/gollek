package tech.kayys.gollek.engine.routing;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;

// API imports
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelManifest;
// Additional imports for model classes
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.provider.RoutingDecision;
import tech.kayys.gollek.engine.context.DevicePreferenceResolver;
import tech.kayys.gollek.model.core.HardwareDetector;
import tech.kayys.gollek.observability.AdapterRoutingMetricsCollector;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.engine.module.SystemModule.RequestConfigRepository;
import tech.kayys.gollek.registry.repository.CachedModelRepository;
import tech.kayys.gollek.engine.routing.policy.SelectionPolicy;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.ModelException;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.metrics.RuntimeMetricsCache;
import tech.kayys.gollek.provider.core.routing.FormatAwareProviderRouter;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCandidate;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.registry.LocalModelRegistry;
import tech.kayys.gollek.spi.registry.ModelEntry;

/**
 * Intelligent model router with multi-factor scoring.
 * Selects optimal provider based on:
 * - Model compatibility
 * - Hardware availability
 * - Historical performance
 * - Cost optimization
 * - Current load
 * - Tenant preferences
 */
@ApplicationScoped
public class ModelRouterService {

        private static final Logger LOG = Logger.getLogger(ModelRouterService.class);

        @Inject
        ProviderRegistry providerRegistry;

        @Inject
        CachedModelRepository modelRepository;

        @Inject
        SelectionPolicy selectionPolicy;

        @Inject
        RuntimeMetricsCache metricsCache;

        @Inject
        AdapterRoutingMetricsCollector adapterRoutingMetricsCollector;

        @Inject
        DevicePreferenceResolver devicePreferenceResolver;

        @Inject
        HardwareDetector hardwareDetector;

        @Inject
        RequestConfigRepository requestConfigRepository;

        @Inject
        tech.kayys.gollek.engine.config.ModelConfig modelConfig;

        @Inject
        FormatAwareProviderRouter formatRouter;

        @Inject
        LocalModelRegistry localModelRegistry;

        private final Map<String, RoutingDecision> decisionCache = new ConcurrentHashMap<>();

        /**
         * Route inference request to optimal provider.
         *
         * <p>
         * If the model resolves to a known local format (GGUF or SafeTensors)
         * via the {@link LocalModelRegistry}, the request is dispatched through
         * {@link FormatAwareProviderRouter} — bypassing manifest lookup and
         * multi-provider
         * scoring. Otherwise the existing multi-factor routing pipeline is used.
         */
        public Uni<InferenceResponse> route(
                        String modelId,
                        InferenceRequest request) {

                String effectiveModelId = (modelId == null || modelId.isBlank())
                                ? modelConfig.defaultModel().orElse(modelId)
                                : modelId;

                if (effectiveModelId == null || effectiveModelId.isBlank()) {
                        return Uni.createFrom().failure(new ModelException(
                                        ErrorCode.MODEL_NOT_FOUND,
                                        "No model specified and no default-model configured", modelId));
                }

                // v0.1.4 — fast path for locally-served models
                if (isLocalModel(effectiveModelId)) {
                        LOG.debugf("ModelRouterService: local route [%s]", effectiveModelId);
                        ProviderRequest pr = toLocalProviderRequest(effectiveModelId, request);
                        return formatRouter.route(pr)
                                        .invoke(resp -> metricsCache.recordSuccess("local", effectiveModelId,
                                                        resp.getDurationMs()))
                                        .onFailure().invoke(err -> metricsCache.recordFailure("local", effectiveModelId,
                                                        err.getClass().getSimpleName()));
                }
                return modelRepository.findById(effectiveModelId, getTenantId(request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createDirectPathManifest(modelId, request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createVirtualProviderManifest(modelId, request))
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId, modelId))
                                .chain(manifest -> {
                                        // Build routing context
                                        RoutingContext context = buildRoutingContext(
                                                        request,
                                                        manifest);

                                        // Select provider
                                        RoutingDecision decision = selectProvider(manifest, context);

                                        // Cache decision for debugging
                                        decisionCache.put(request.getRequestId(), decision);

                                        LOG.infof("Routing model %s to provider %s (score: %d)",
                                                        modelId, decision.providerId(), decision.score());

                                        return executeWithProvider(decision, request);
                                })
                                .onFailure().retry().withBackOff(Duration.ofMillis(100))
                                .atMost(3)
                                .onFailure()
                                .recoverWithUni(error -> handleRoutingFailure(modelId, request, error));
        }

        /**
         * Route streaming inference request to optimal provider
         */
        public Multi<StreamingInferenceChunk> routeStream(
                        String modelId,
                        InferenceRequest request) {

                String effectiveModelId = (modelId == null || modelId.isBlank())
                                ? modelConfig.defaultModel().orElse(modelId)
                                : modelId;

                if (effectiveModelId == null || effectiveModelId.isBlank()) {
                        return Multi.createFrom().failure(new ModelException(
                                        ErrorCode.MODEL_NOT_FOUND,
                                        "No model specified and no default-model configured", modelId));
                }

                // v0.1.4 — fast path for locally-served models
                if (isLocalModel(effectiveModelId)) {
                        LOG.debugf("ModelRouterService: local stream route [%s]", effectiveModelId);
                        ProviderRequest pr = toLocalProviderRequest(effectiveModelId, request);
                        return formatRouter.routeStream(pr)
                                        .onFailure().invoke(err -> metricsCache.recordFailure("local", effectiveModelId,
                                                        err.getClass().getSimpleName()));
                }

                return modelRepository.findById(effectiveModelId, getTenantId(request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createDirectPathManifest(modelId, request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createVirtualProviderManifest(modelId, request))
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId, modelId))
                                .onItem().transformToMulti(manifest -> {
                                        RoutingContext context = buildRoutingContext(request, manifest);
                                        RoutingDecision decision = selectProvider(manifest, context);
                                        return executeStreamWithProvider(decision, request);
                                });
        }

        /**
         * Route embedding request to optimal provider
         */
        public Uni<tech.kayys.gollek.spi.embedding.EmbeddingResponse> routeEmbedding(
                        String modelId,
                        tech.kayys.gollek.spi.embedding.EmbeddingRequest request) {

                String effectiveModelId = (modelId == null || modelId.isBlank())
                                ? modelConfig.defaultModel().orElse(modelId)
                                : modelId;

                // v0.1.4 — local embedding fast path
                if (isLocalModel(effectiveModelId)) {
                        // v0.1.4 — local embedding: log the local hit but fall through
                        // to the existing provider pipeline, which already handles embedding
                        LOG.debugf("ModelRouterService: local embed detected [%s]", effectiveModelId);
                }
                return modelRepository.findById(effectiveModelId, "community") // Default tenant for embedding for now
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + effectiveModelId, effectiveModelId))
                                .chain(manifest -> {
                                        // Select provider with embedding capability
                                        List<LLMProvider> providers = providerRegistry.getAllProviders().stream()
                                                        .filter(p -> p.capabilities().isEmbeddings())
                                                        .toList();

                                        if (providers.isEmpty()) {
                                                throw new ProviderException(
                                                                "No provider with embedding capability found");
                                        }

                                        // For now, take first compatible or use simple scoring
                                        LLMProvider provider = providers.get(0);

                                        EmbeddingRequest embeddingRequest = new EmbeddingRequest(
                                                        request.requestId(),
                                                        manifest.modelId(),
                                                        request.inputs(),
                                                        request.parameters());

                                        return provider.embed(embeddingRequest);
                                });
        }

        private String getTenantId(InferenceRequest request) {
                return request.getMetadata().getOrDefault("tenantId", "community").toString();
        }

        private ModelManifest createVirtualProviderManifest(String modelId, InferenceRequest request) {
                if (request.getPreferredProvider().isPresent()) {
                        String preferred = request.getPreferredProvider().get();
                        if (providerRegistry.hasProvider(preferred)) {
                                return ModelManifest.builder()
                                                .modelId(modelId)
                                                .name(modelId)
                                                .version("latest")
                                                .path("virtual")
                                                .apiKey(request.getApiKey())
                                                .requestId(request.getRequestId())
                                                .metadata(Map.of("source", "virtual-provider", "provider", preferred))
                                                .artifacts(Map.of())
                                                .createdAt(Instant.now())
                                                .updatedAt(Instant.now())
                                                .build();
                        }
                }
                return null;
        }

        private ModelManifest createDirectPathManifest(String modelId, InferenceRequest request) {
                Object requestedPath = request.getParameters().get("model_path");
                String candidate = requestedPath instanceof String s && !s.isBlank() ? s : modelId;

                try {
                        Path modelPath = Path.of(candidate).toAbsolutePath().normalize();
                        if (!Files.exists(modelPath) || !Files.isRegularFile(modelPath)) {
                                return null;
                        }

                        // Detect format from extension
                        String filename = modelPath.getFileName().toString().toLowerCase();
                        ModelFormat format = ModelFormat.GGUF; // Default
                        if (filename.endsWith(".litertlm") || filename.endsWith(".tflite") || filename.endsWith(".task")) {
                            format = ModelFormat.LITERT;
                        } else if (filename.endsWith(".safetensors")) {
                            format = ModelFormat.SAFETENSORS;
                        }

                        return ModelManifest.builder()
                                        .modelId(modelPath.toString())
                                        .name(modelPath.getFileName().toString())
                                        .version("local")
                                        .path(modelPath.toString())
                                        .apiKey(request.getApiKey())
                                        .requestId(request.getRequestId())
                                        .artifacts(Map.of(format,
                                                        new tech.kayys.gollek.spi.model.ArtifactLocation(
                                                                        modelPath.toString(), null, null, null)))
                                        .resourceRequirements(null)
                                        .metadata(Map.of("source", "direct-path"))
                                        .createdAt(Instant.now())
                                        .updatedAt(Instant.now())
                                        .build();
                } catch (Exception e) {
                        LOG.debugf("Direct model path fallback rejected for candidate '%s': %s", candidate,
                                        e.getMessage());
                        return null;
                }
        }

        /**
         * Select best provider using multi-factor scoring
         */
        private RoutingDecision selectProvider(
                        ModelManifest manifest,
                        RoutingContext context) {
                // Get all available providers
                List<LLMProvider> providers = new java.util.ArrayList<>(providerRegistry.getAllProviders());
                LOG.debugf("Selecting provider for model %s from %d providers", manifest.modelId(), providers.size());

                // Use ProviderRequest for support check (needs translation from
                // InferenceRequest/Context)
                // ideally LLMProvider.supports takes InferenceRequest or compatible
                // Current LLMProvider.supports takes ProviderRequest.
                // We need to construct a lightweight ProviderRequest or update supports to take
                // InferenceRequest
                // For now, let's construct a minimal ProviderRequest
                ProviderRequest checkRequest = ProviderRequest.builder()
                                .model(manifest.modelId())
                                .messages(context.request().getMessages())
                                .parameters(context.request().getParameters())
                                .metadata(context.request().getMetadata())
                                .metadata("tenantId", getTenantId(context.request()))
                                .build();

                // Explicit provider pinning from request should be honored strictly.
                if (context.preferredProvider().isPresent()) {
                        String preferred = context.preferredProvider().get();
                        Optional<LLMProvider> pinned = providers.stream()
                                        .filter(p -> p.id().equalsIgnoreCase(preferred))
                                        .filter(p -> p.supports(manifest.modelId(), checkRequest))
                                        .findFirst();
                        if (pinned.isPresent()) {
                                LLMProvider provider = pinned.get();
                                return RoutingDecision.builder()
                                                .providerId(provider.id())
                                                .provider(provider)
                                                .score(10_000)
                                                .fallbackProviders(java.util.List.of())
                                                .manifest(manifest)
                                                .context(context)
                                                .build();
                        }
                }

                // Filter compatible providers
                List<ProviderCandidate> candidates = providers.stream()
                                .filter(p -> isFormatCompatible(p, manifest))
                                .filter(p -> isAdapterCompatible(p, checkRequest))
                                .filter(p -> p.supports(manifest.modelId(), checkRequest))
                                .map(p -> scoreProvider(p, manifest, context))
                                .filter(Optional::isPresent)
                                .map(Optional::get)
                                .collect(Collectors.toList());

                if (candidates.isEmpty()) {
                        Optional<ProviderCandidate> ggufFallback = tryGgufFallback(providers, manifest, context,
                                        checkRequest);
                        if (ggufFallback.isPresent()) {
                                ProviderCandidate fallback = ggufFallback.get();
                                LOG.warnf("Using GGUF fallback provider for model %s", manifest.modelId());
                                return RoutingDecision.builder()
                                                .providerId(fallback.providerId())
                                                .provider(fallback.provider())
                                                .score(fallback.score())
                                                .fallbackProviders(java.util.List.of())
                                                .manifest(manifest)
                                                .context(context)
                                                .build();
                        }
                        LOG.warnf("No compatible provider found for model %s. Available providers: %s",
                                        manifest.modelId(),
                                        providers.stream().map(LLMProvider::id).collect(Collectors.joining(", ")));
                        throw new ProviderException(
                                        "No compatible provider found for model: " + manifest.modelId());
                }

                // Honor explicit preferred provider when it is compatible.
                if (context.preferredProvider().isPresent()) {
                        String preferred = context.preferredProvider().get();
                        Optional<ProviderCandidate> preferredCandidate = candidates.stream()
                                        .filter(c -> c.providerId().equalsIgnoreCase(preferred))
                                        .findFirst();
                        if (preferredCandidate.isPresent()) {
                                ProviderCandidate selected = preferredCandidate.get();
                                return RoutingDecision.builder()
                                                .providerId(selected.providerId())
                                                .provider(selected.provider())
                                                .score(selected.score())
                                                .fallbackProviders(candidates.stream()
                                                                .filter(c -> !c.providerId()
                                                                                .equalsIgnoreCase(
                                                                                                selected.providerId()))
                                                                .limit(2)
                                                                .map(ProviderCandidate::providerId)
                                                                .collect(Collectors.toList()))
                                                .manifest(manifest)
                                                .context(context)
                                                .build();
                        }
                }

                // Sort by score descending
                candidates.sort(Comparator.comparing(
                                ProviderCandidate::score).reversed());

                // Select top candidate
                ProviderCandidate winner = candidates.get(0);

                LOG.debugf("Provider selection for %s: %s (score: %d), alternatives: %d",
                                manifest.modelId(),
                                winner.providerId(),
                                winner.score(),
                                candidates.size() - 1);

                return RoutingDecision.builder()
                                .providerId(winner.providerId())
                                .provider(winner.provider())
                                .score(winner.score())
                                .fallbackProviders(candidates.stream()
                                                .skip(1)
                                                .limit(2)
                                                .map(ProviderCandidate::providerId)
                                                .collect(Collectors.toList()))
                                .manifest(manifest)
                                .context(context)
                                .build();
        }

        private boolean isFormatCompatible(LLMProvider provider, ModelManifest manifest) {
                Set<ModelFormat> providerFormats = provider.capabilities().getSupportedFormats();
                if (providerFormats == null || providerFormats.isEmpty()) {
                        return true; // Assume generic provider if no format restrictions
                }
                return manifest.artifacts().keySet().stream()
                                .anyMatch(providerFormats::contains);
        }

        private boolean isAdapterCompatible(LLMProvider provider, ProviderRequest request) {
                if (!AdapterRoutingSupport.hasAdapterRequest(request)) {
                        return true;
                }

                ProviderCapabilities capabilities = provider.capabilities();
                if (AdapterRoutingSupport.isAdapterUnsupported(capabilities)) {
                        LOG.debugf("Skipping provider %s for adapter request due to adapter_unsupported capability",
                                        provider.id());
                        if (adapterRoutingMetricsCollector != null) {
                                String tenantId = String
                                                .valueOf(request.getMetadata().getOrDefault("tenantId", "community"));
                                adapterRoutingMetricsCollector.recordProviderFiltered(
                                                "model-router-service",
                                                provider.id(),
                                                request.getModel(),
                                                tenantId,
                                                "adapter_unsupported");
                        }
                        return false;
                }
                return true;
        }

        private Optional<ProviderCandidate> scoreProvider(
                        LLMProvider provider,
                        ModelManifest manifest,
                        RoutingContext context) {
                int score = 50; // Baseline

                ProviderCapabilities caps = provider.capabilities();

                // 1. Streaming support match
                if (caps.isStreaming() && context.request().isStreaming()) {
                        score += 20;
                }
                if (!caps.isStreaming() && context.request().isStreaming()) {
                        score -= 15;
                }

                // Strong preference for user/provider-pinned routing.
                if (context.preferredProvider().isPresent()) {
                        if (provider.id().equalsIgnoreCase(context.preferredProvider().get())) {
                                score += 1_000;
                        } else {
                                score -= 100;
                        }
                }

                // 2. Device preference match
                if (context.requestContext() != null && context.requestContext().preferredDevice().isPresent()) {
                        DeviceType preferred = context.requestContext().preferredDevice().get();
                        if (caps.getSupportedDevices().contains(preferred)) {
                                score += 30;
                        }
                }

                // 3. Cost sensitivity
                if (context.costSensitive() && caps.getSupportedDevices().contains(DeviceType.CPU)) {
                        score += 10;
                }

                return Optional.of(new ProviderCandidate(
                                provider.id(),
                                provider,
                                score,
                                Duration.ZERO,
                                0.0));
        }

        private Optional<ProviderCandidate> tryGgufFallback(
                        List<LLMProvider> providers,
                        ModelManifest manifest,
                        RoutingContext context,
                        ProviderRequest checkRequest) {
                return providers.stream()
                                .filter(p -> isAdapterCompatible(p, checkRequest))
                                .filter(p -> p.id().toLowerCase().contains("gguf")
                                                || p.id().toLowerCase().contains("llama"))
                                .map(p -> new ProviderCandidate(p.id(), p, 40, Duration.ZERO, 0.0))
                                .findFirst();
        }

        private Uni<InferenceResponse> executeWithProvider(
                        RoutingDecision decision,
                        InferenceRequest request) {
                LLMProvider provider = decision.provider();

                // Build provider request
                ProviderRequest providerRequest = ProviderRequest.builder()
                                .model(decision.manifest().modelId())
                                .messages(request.getMessages())
                                .parameters(request.getParameters())
                                .streaming(request.isStreaming())
                                .timeout(decision.context().timeout())
                                .metadata("request_id", request.getRequestId())
                                .metadata("tenantId", getTenantId(request))
                                .build();

                return provider.infer(providerRequest)
                                .onItem().invoke(response -> {
                                        // Record metrics
                                        metricsCache.recordSuccess(
                                                        provider.id(),
                                                        decision.manifest().modelId(),
                                                        response.getDurationMs());
                                })
                                .onFailure().invoke(error -> {
                                        // Record failure
                                        metricsCache.recordFailure(
                                                        provider.id(),
                                                        decision.manifest().modelId(),
                                                        error.getClass().getSimpleName());
                                });
        }

        private Multi<StreamingInferenceChunk> executeStreamWithProvider(
                        RoutingDecision decision,
                        InferenceRequest request) {
                LLMProvider provider = decision.provider();

                if (!(provider instanceof StreamingProvider streamingProvider)) {
                        return Multi.createFrom().failure(new UnsupportedOperationException(
                                        "Provider " + provider.id() + " does not support streaming"));
                }

                ProviderRequest.Builder requestBuilder = ProviderRequest.builder()
                                .model(decision.manifest().modelId())
                                .messages(request.getMessages())
                                .parameters(request.getParameters())
                                .streaming(true)
                                .timeout(decision.context().timeout())
                                .metadata("request_id", request.getRequestId())
                                .metadata("tenantId", getTenantId(request));

                return streamingProvider.inferStream(requestBuilder.build())
                                .onFailure().invoke(error -> {
                                        metricsCache.recordFailure(
                                                        provider.id(),
                                                        decision.manifest().modelId(),
                                                        error.getClass().getSimpleName());
                                });
        }

        private Uni<InferenceResponse> handleRoutingFailure(
                        String modelId,
                        InferenceRequest request,
                        Throwable error) {
                LOG.errorf(error, "Routing failed for model %s", modelId);

                // Try fallback providers if available
                RoutingDecision lastDecision = decisionCache.get(request.getRequestId());

                if (lastDecision != null && !lastDecision.fallbackProviders().isEmpty()) {
                        String fallbackId = lastDecision.fallbackProviders().get(0);

                        LOG.infof("Attempting fallback to provider: %s", fallbackId);

                        return providerRegistry.getProvider(fallbackId)
                                        .map(provider -> {
                                                ProviderRequest providerRequest = ProviderRequest.builder()
                                                                .model(modelId)
                                                                .messages(request.getMessages())
                                                                .parameters(request.getParameters())
                                                                .metadata("tenantId", getTenantId(request))
                                                                .build();

                                                return provider.infer(providerRequest);
                                        })
                                        .orElseGet(() -> Uni.createFrom().failure(error));
                }

                return Uni.createFrom().failure(error);
        }

        private RoutingContext buildRoutingContext(
                        InferenceRequest request,
                        ModelManifest manifest) {
                Duration timeout = request.getTimeout()
                                .orElse(Duration.ofSeconds(30));

                String tenantId = getTenantId(request);
                RequestContext context = RequestContext.forTenant(tenantId, request.getRequestId());
                context = devicePreferenceResolver.apply(context, request);

                String preferredProvider = request.getPreferredProvider()
                                .orElseGet(() -> modelConfig.defaultProvider());

                return RoutingContext.builder()
                                .request(request)
                                .requestContext(context)
                                .preferredProvider(preferredProvider)
                                .deviceHint(extractDeviceHint(request).orElse(null))
                                .timeout(timeout)
                                .costSensitive(isCostSensitive(request, context))
                                .priority(request.getPriority())
                                .build();
        }

        private Optional<String> extractDeviceHint(InferenceRequest request) {
                return request.getParameters().containsKey("device")
                                ? Optional.of(request.getParameters().get("device").toString())
                                : Optional.empty();
        }

        private boolean isCostSensitive(
                        InferenceRequest request,
                        RequestContext context) {
                if (context != null && context.isCostSensitive()) {
                        return true;
                }
                // Hook for enterprise configuration. Community defaults to false.
                return requestConfigRepository != null
                                && requestConfigRepository.isCostSensitive(context.getRequestId());
        }

        // ── v0.1.4 — Local model helpers ──────────────────────────────────────

        /**
         * Determine whether the model should be served by a local provider.
         *
         * <p>
         * A model is considered local when:
         * <ul>
         * <li>The {@link LocalModelRegistry} resolves it to a known GGUF or
         * SafeTensors entry, or</li>
         * <li>{@link FormatAwareProviderRouter#resolveFormat} returns GGUF or
         * SAFETENSORS.</li>
         * </ul>
         */
        private boolean isLocalModel(String modelId) {
                // Fast path: registry hit
                Optional<ModelEntry> entry = localModelRegistry.resolve(modelId);
                if (entry.isPresent()) {
                        ModelFormat fmt = entry.get().format();
                        return fmt == ModelFormat.GGUF || fmt == ModelFormat.SAFETENSORS || fmt == ModelFormat.LITERT;
                }

                // Detect by path / extension
                Optional<ModelFormat> detected = formatRouter.resolveFormat(modelId);
                return detected.map(f -> f == ModelFormat.GGUF || f == ModelFormat.SAFETENSORS || f == ModelFormat.LITERT)
                                .orElse(false);
        }

        /**
         * Convert an engine-level {@link InferenceRequest} to a provider-level
         * {@link ProviderRequest} for local model routing.
         */
        private ProviderRequest toLocalProviderRequest(String modelId, InferenceRequest request) {
                String tenantId = request.getMetadata() != null
                                ? request.getMetadata().getOrDefault("tenantId", "community").toString()
                                : "community";

                return ProviderRequest.builder()
                                .requestId(request.getRequestId())
                                .model(modelId)
                                .messages(request.getMessages())
                                .parameters(request.getParameters())
                                .tools(request.getTools())
                                .toolChoice(request.getToolChoice())
                                .streaming(request.isStreaming())
                                .timeout(request.getTimeout().orElse(null))
                                .metadata("tenantId", tenantId)
                                .build();
        }

        /**
         * Get routing decision for debugging
         */
        public Optional<RoutingDecision> getLastDecision(String requestId) {
                return Optional.ofNullable(decisionCache.get(requestId));
        }

        /**
         * Clear decision cache
         */
        public void clearDecisionCache() {
                decisionCache.clear();
        }
}
