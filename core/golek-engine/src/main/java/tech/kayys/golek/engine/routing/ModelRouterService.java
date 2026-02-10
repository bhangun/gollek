package tech.kayys.golek.engine.routing;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.stream.StreamChunk;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.spi.provider.StreamingProvider;
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

// API imports
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.wayang.tenant.TenantConfigRepository;
// Core engine imports

import tech.kayys.golek.core.exception.NoCompatibleProviderException;

// Additional imports for model classes
import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ResourceRequirements;
import tech.kayys.golek.spi.provider.RoutingContext;
import tech.kayys.golek.spi.provider.RoutingDecision;
import tech.kayys.golek.spi.provider.ProviderCandidate;
import tech.kayys.golek.engine.observability.RuntimeMetricsCache;
import tech.kayys.golek.model.core.HardwareCapabilities;
import tech.kayys.golek.model.core.HardwareDetector;
import tech.kayys.golek.spi.exception.ModelException;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.engine.model.CachedModelRepository;
import tech.kayys.golek.spi.provider.ProviderRegistry;
import tech.kayys.golek.engine.routing.policy.SelectionPolicy;
import tech.kayys.golek.spi.provider.LLMProvider;

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
    TenantConfigRepository tenantConfigRepository;

    private final Map<String, RoutingDecision> decisionCache = new ConcurrentHashMap<>();

    /**
     * Route inference request to optimal provider
     */
    public Uni<InferenceResponse> route(
            String modelId,
            InferenceRequest request,
            TenantContext tenantContext) {
        TenantContext effectiveTenantContext = ensureTenantContext(tenantContext);

        return modelRepository.findById(modelId, effectiveTenantContext.getTenantId().value())
                .onItem().ifNull().failWith(() -> new ModelException(
                        tech.kayys.golek.spi.error.ErrorCode.MODEL_NOT_FOUND,
                        "Model not found: " + modelId, modelId))
                .chain(manifest -> {
                    // Build routing context
                    RoutingContext context = buildRoutingContext(
                            request,
                            effectiveTenantContext,
                            manifest);

                    // Select provider
                    RoutingDecision decision = selectProvider(manifest, context);

                    // Cache decision for debugging
                    decisionCache.put(request.getRequestId(), decision);

                    LOG.infof("Routing model %s to provider %s (score: %d)",
                            modelId, decision.providerId(), decision.score());

                    return executeWithProvider(decision, request, effectiveTenantContext);
                })
                .onFailure().retry().withBackOff(Duration.ofMillis(100))
                .atMost(3)
                .onFailure()
                .recoverWithUni(error -> handleRoutingFailure(modelId, request, effectiveTenantContext, error));
    }

    /**
     * Route streaming inference request to optimal provider
     */
    public Multi<StreamChunk> routeStream(
            String modelId,
            InferenceRequest request,
            TenantContext tenantContext) {
        TenantContext effectiveTenantContext = ensureTenantContext(tenantContext);

        return modelRepository.findById(modelId, effectiveTenantContext.getTenantId().value())
                .onItem().ifNull().failWith(() -> new ModelException(
                        tech.kayys.golek.spi.error.ErrorCode.MODEL_NOT_FOUND,
                        "Model not found: " + modelId, modelId))
                .onItem().transformToMulti(manifest -> {
                    RoutingContext context = buildRoutingContext(request, effectiveTenantContext, manifest);
                    RoutingDecision decision = selectProvider(manifest, context);
                    return executeStreamWithProvider(decision, request, effectiveTenantContext);
                });
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
        for (LLMProvider p : providers) {
            boolean supported = p.supports(manifest.modelId(), context.tenantContext());
            LOG.debugf("Provider %s support for %s: %b", p.id(), manifest.modelId(), supported);
        }

        // Filter compatible providers
        List<ProviderCandidate> candidates = providers.stream()
                .filter(p -> p.supports(manifest.modelId(), context.tenantContext()))
                .map(p -> scoreProvider(p, manifest, context))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());

        if (candidates.isEmpty()) {
            LOG.warnf("No compatible provider found for model %s. Available providers: %s",
                    manifest.modelId(),
                    providers.stream().map(LLMProvider::id).collect(Collectors.joining(", ")));
            throw new NoCompatibleProviderException(
                    "No compatible provider found for model: " + manifest.modelId());
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

    /**
     * Score provider using multi-factor algorithm
     */
    private Optional<ProviderCandidate> scoreProvider(
            LLMProvider provider,
            ModelManifest manifest,
            RoutingContext context) {
        try {
            int score = 0;

            // 1. Preferred provider match (highest weight: 100)
            if (context.preferredProvider().isPresent() &&
                    provider.id().equals(context.preferredProvider().get())) {
                score += 100;
            }

            // 2. Native format support (weight: 50)
            if (supportsNativeFormat(provider, manifest)) {
                score += 50;
            }

            // 3. Device availability & preference (weight: 40)
            score += scoreDeviceCompatibility(provider, context);

            // 4. Historical performance (weight: 30)
            score += scorePerformance(provider, manifest, context);

            // 5. Current availability (weight: 25)
            score += scoreAvailability(provider);

            // 6. Cost optimization (weight: 20)
            score += scoreCost(provider, context);

            // 7. Feature compatibility (weight: 15)
            score += scoreFeatures(provider, context.request());

            // 8. Load balancing (weight: 10)
            score += scoreLoadBalance(provider);

            // Penalties
            score -= calculatePenalties(provider, manifest, context);

            LOG.tracef("Provider %s scored %d for model %s",
                    provider.id(), score, manifest.modelId());

            return Optional.of(new ProviderCandidate(
                    provider.id(),
                    provider,
                    score,
                    calculateEstimatedLatency(provider, manifest),
                    calculateEstimatedCost(provider, manifest, context)));

        } catch (Exception e) {
            LOG.warnf(e, "Failed to score provider %s", provider.id());
            return Optional.empty();
        }
    }

    private boolean supportsNativeFormat(
            LLMProvider provider,
            ModelManifest manifest) {
        Set<ModelFormat> supportedFormats = provider.capabilities()
                .getSupportedFormats();

        return manifest.artifacts().keySet().stream()
                .anyMatch(supportedFormats::contains);
    }

    private int scoreDeviceCompatibility(
            LLMProvider provider,
            RoutingContext context) {
        HardwareCapabilities hw = hardwareDetector.detect();
        Set<DeviceType> supportedDevices = provider.capabilities()
                .getSupportedDevices();

        // Preferred device match
        if (context.deviceHint().isPresent()) {
            try {
                DeviceType hint = DeviceType.valueOf(context.deviceHint().get().toUpperCase());
                if (supportedDevices.contains(hint)) {
                    return 40;
                }
            } catch (IllegalArgumentException ignored) {
            }
        }

        // CUDA available and supported
        if (hw.hasCUDA() && supportedDevices.contains(DeviceType.CUDA)) {
            return 30;
        }

        // CPU fallback
        if (supportedDevices.contains(DeviceType.CPU)) {
            return 10;
        }

        return 0;
    }

    private int scorePerformance(
            LLMProvider provider,
            ModelManifest manifest,
            RoutingContext context) {
        // Get P95 latency from metrics
        Optional<Duration> p95 = metricsCache.getP95Latency(
                provider.id(),
                manifest.modelId());

        if (p95.isEmpty()) {
            return 0; // No historical data
        }

        Duration timeout = context.timeout();

        // Within timeout with margin
        if (p95.get().compareTo(timeout.multipliedBy(2).dividedBy(3)) < 0) {
            return 30;
        }

        // Barely within timeout
        if (p95.get().compareTo(timeout) < 0) {
            return 15;
        }

        // Exceeds timeout
        return -20;
    }

    private int scoreAvailability(LLMProvider provider) {
        try {
            ProviderHealth health = provider.health().await().atMost(Duration.ofMillis(1000));
            if (!health.isHealthy()) {
                return -50; // Heavy penalty for unhealthy
            }
        } catch (Exception e) {
            LOG.warnf("Health check failed for provider %s: %s", provider.id(), e.getMessage());
            return 0; // Neutral score on failure
        }

        // Check recent error rate
        double errorRate = metricsCache.getErrorRate(
                provider.id(),
                Duration.ofMinutes(5));

        if (errorRate < 0.01) { // < 1%
            return 25;
        } else if (errorRate < 0.05) { // < 5%
            return 10;
        } else if (errorRate < 0.10) { // < 10%
            return -10;
        } else {
            return -30;
        }
    }

    private int scoreCost(LLMProvider provider, RoutingContext context) {
        if (!context.costSensitive()) {
            return 0;
        }

        // Prefer local/free providers when cost-sensitive
        if (provider.id().contains("local") ||
                provider.id().contains("onnx") ||
                provider.id().contains("pytorch")) {
            return 20;
        }

        // Cloud providers cost more
        if (provider.id().contains("openai") ||
                provider.id().contains("anthropic")) {
            return -10;
        }

        return 0;
    }

    private int scoreFeatures(
            LLMProvider provider,
            InferenceRequest request) {
        int score = 0;
        ProviderCapabilities caps = provider.capabilities();

        // Streaming support
        if (request.isStreaming() && caps.isStreaming()) {
            score += 15;
        } else if (request.isStreaming() && !caps.isStreaming()) {
            return -50; // Cannot fulfill requirement
        }

        // Function calling
        if (requiresFunctionCalling(request) && caps.isFunctionCalling()) {
            score += 10;
        } else if (requiresFunctionCalling(request)) {
            return -50;
        }

        return score;
    }

    private int scoreLoadBalance(LLMProvider provider) {
        double currentLoad = metricsCache.getCurrentLoad(provider.id());

        if (currentLoad < 0.3) {
            return 10; // Low load
        } else if (currentLoad < 0.7) {
            return 5; // Medium load
        } else if (currentLoad < 0.9) {
            return -5; // High load
        } else {
            return -20; // Overloaded
        }
    }

    private int calculatePenalties(
            LLMProvider provider,
            ModelManifest manifest,
            RoutingContext context) {
        int penalties = 0;

        // Resource constraint violations
        if (!hasAvailableResources(provider, manifest, context)) {
            penalties += 30;
        }

        // Quota exhausted
        if (isQuotaExhausted(provider, context.tenantContext())) {
            penalties += 100; // Severe penalty
        }

        // Circuit breaker open
        if (isCircuitBreakerOpen(provider)) {
            penalties += 100;
        }

        return penalties;
    }

    private boolean hasAvailableResources(
            LLMProvider provider,
            ModelManifest manifest,
            RoutingContext context) {
        HardwareCapabilities hw = hardwareDetector.detect();

        // Resource checks
        ResourceRequirements requirements = manifest.resourceRequirements();
        if (requirements != null && requirements.memory() != null) {
            Long requiredMemoryMb = requirements.memory().minMemoryMb();
            if (requiredMemoryMb != null && hw.getAvailableMemory() < (requiredMemoryMb * 1024 * 1024)) {
                return false;
            }
        }

        return true;
    }

    private boolean isQuotaExhausted(
            LLMProvider provider,
            TenantContext context) {
        // Check tenant quota
        return tenantConfigRepository.isQuotaExhausted(
                ensureTenantContext(context).getTenantId().value(),
                provider.id());
    }

    private boolean isCircuitBreakerOpen(LLMProvider provider) {
        return metricsCache.isCircuitBreakerOpen(provider.id());
    }

    private boolean requiresFunctionCalling(InferenceRequest request) {
        return request.getParameters().containsKey("functions") ||
                request.getParameters().containsKey("tools");
    }

    private Duration calculateEstimatedLatency(
            LLMProvider provider,
            ModelManifest manifest) {
        return metricsCache.getP95Latency(provider.id(), manifest.modelId())
                .orElse(Duration.ofSeconds(5));
    }

    private double calculateEstimatedCost(
            LLMProvider provider,
            ModelManifest manifest,
            RoutingContext context) {
        // Simplified cost calculation
        int estimatedTokens = context.request().getMessages().stream()
                .mapToInt(m -> m.getContent().split("\\s+").length)
                .sum();

        return estimatedTokens * 0.00001; // $0.01 per 1K tokens
    }

    private Uni<InferenceResponse> executeWithProvider(
            RoutingDecision decision,
            InferenceRequest request,
            TenantContext context) {
        LLMProvider provider = decision.provider();

        // Build provider request
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model(decision.manifest().modelId())
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(request.isStreaming())
                .timeout(decision.context().timeout())
                .metadata("request_id", request.getRequestId())
                .metadata("tenant_id", context.getTenantId())
                .build();

        return provider.infer(providerRequest, context)
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
            InferenceRequest request,
            TenantContext context) {
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
                .metadata("tenant_id", context.getTenantId());

        return streamingProvider.inferStream(
                requestBuilder.build(),
                context)
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
            TenantContext context,
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
                                .build();

                        return provider.infer(providerRequest, context);
                    })
                    .orElseGet(() -> Uni.createFrom().failure(error));
        }

        return Uni.createFrom().failure(error);
    }

    private RoutingContext buildRoutingContext(
            InferenceRequest request,
            TenantContext tenantContext,
            ModelManifest manifest) {
        Duration timeout = request.getTimeout()
                .orElse(Duration.ofSeconds(30));

        return RoutingContext.builder()
                .request(request)
                .tenantContext(tenantContext)
                .preferredProvider(request.getPreferredProvider().orElse(null))
                .deviceHint(extractDeviceHint(request).orElse(null))
                .timeout(timeout)
                .costSensitive(isCostSensitive(request, tenantContext))
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
            TenantContext context) {
        // Check if tenant has cost-sensitive flag
        return tenantConfigRepository.isCostSensitive(ensureTenantContext(context).getTenantId().value());
    }

    private TenantContext ensureTenantContext(TenantContext tenantContext) {
        return tenantContext != null ? tenantContext : TenantContext.of("community");
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
