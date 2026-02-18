package tech.kayys.gollek.engine.routing;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.stream.StreamChunk;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.provider.StreamingProvider;
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
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRequest;
// Core engine imports

import tech.kayys.gollek.core.exception.NoCompatibleProviderException;

// Additional imports for model classes
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.provider.RoutingDecision;
import tech.kayys.gollek.spi.provider.ProviderCandidate;
import tech.kayys.gollek.engine.observability.RuntimeMetricsCache;
import tech.kayys.gollek.model.core.HardwareDetector;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.exception.ModelException;
import tech.kayys.gollek.engine.module.SystemModule.RequestConfigRepository;
import tech.kayys.gollek.engine.model.CachedModelRepository;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.engine.routing.policy.SelectionPolicy;
import tech.kayys.gollek.spi.provider.LLMProvider;

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
        HardwareDetector hardwareDetector;

        @Inject
        RequestConfigRepository requestConfigRepository;

        private final Map<String, RoutingDecision> decisionCache = new ConcurrentHashMap<>();

        /**
         * Route inference request to optimal provider
         */
        public Uni<InferenceResponse> route(
                        String modelId,
                        InferenceRequest request) {

                return modelRepository.findById(modelId, getTenantId(request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createDirectPathManifest(modelId, request))
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                tech.kayys.gollek.spi.error.ErrorCode.MODEL_NOT_FOUND,
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
        public Multi<StreamChunk> routeStream(
                        String modelId,
                        InferenceRequest request) {

                return modelRepository.findById(modelId, getTenantId(request))
                                .onItem().transform(manifest -> manifest != null ? manifest
                                                : createDirectPathManifest(modelId, request))
                                .onItem().ifNull().failWith(() -> new ModelException(
                                                tech.kayys.gollek.spi.error.ErrorCode.MODEL_NOT_FOUND,
                                                "Model not found: " + modelId, modelId))
                                .onItem().transformToMulti(manifest -> {
                                        RoutingContext context = buildRoutingContext(request, manifest);
                                        RoutingDecision decision = selectProvider(manifest, context);
                                        return executeStreamWithProvider(decision, request);
                                });
        }

        private String getTenantId(InferenceRequest request) {
                return request.getMetadata().getOrDefault("tenantId", "community").toString();
        }

        private ModelManifest createDirectPathManifest(String modelId, InferenceRequest request) {
                Object requestedPath = request.getParameters().get("model_path");
                String candidate = requestedPath instanceof String s && !s.isBlank() ? s : modelId;

                try {
                        Path modelPath = Path.of(candidate).toAbsolutePath().normalize();
                        if (!Files.exists(modelPath) || !Files.isRegularFile(modelPath)) {
                                return null;
                        }

                        return ModelManifest.builder()
                                        .modelId(modelPath.toString())
                                        .name(modelPath.getFileName().toString())
                                        .version("local")
                                        .path(modelPath.toString())
                                        .apiKey(request.getApiKey())
                                        .requestId(request.getRequestId())
                                        .artifacts(Map.of(ModelFormat.GGUF,
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
                                .metadata("tenantId", getTenantId(context.request()))
                                .build();

                // Explicit provider pinning from request should be honored strictly.
                if (context.preferredProvider().isPresent()) {
                        String preferred = context.preferredProvider().get();
                        Optional<LLMProvider> pinned = providers.stream()
                                        .filter(p -> p.id().equalsIgnoreCase(preferred))
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

                for (LLMProvider p : providers) {
                        boolean supported = p.supports(manifest.modelId(), checkRequest);
                        LOG.debugf("Provider %s support for %s: %b", p.id(), manifest.modelId(), supported);
                }

                // Filter compatible providers
                List<ProviderCandidate> candidates = providers.stream()
                                .filter(p -> isFormatCompatible(p, manifest))
                                .filter(p -> p.supports(manifest.modelId(), checkRequest))
                                .map(p -> scoreProvider(p, manifest, context))
                                .filter(Optional::isPresent)
                                .map(Optional::get)
                                .collect(Collectors.toList());

                if (candidates.isEmpty()) {
                        Optional<ProviderCandidate> ggufFallback = tryGgufFallback(providers, manifest, context);
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
                        throw new NoCompatibleProviderException(
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
                                                                                .equalsIgnoreCase(selected.providerId()))
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
                if (context.requestContext() != null && context.requestContext().getPreferredDevice().isPresent()) {
                        DeviceType preferred = context.requestContext().getPreferredDevice().get();
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
                        RoutingContext context) {
                return providers.stream()
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

        private Multi<StreamChunk> executeStreamWithProvider(
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

                return RoutingContext.builder()
                                .request(request)
                                .requestContext(context)
                                .preferredProvider(request.getPreferredProvider().orElse(null))
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
