package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
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

// API imports
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.tenant.TenantContext;

// Core engine imports
import tech.kayys.golek.core.provider.ProviderRegistry;
import tech.kayys.golek.core.provider.LLMProvider;
import tech.kayys.golek.core.provider.ProviderRequest;
import tech.kayys.golek.core.provider.ProviderCapabilities;
import tech.kayys.golek.core.provider.ProviderHealth;
import tech.kayys.golek.engine.tenant.TenantConfigRepository;
import tech.kayys.golek.model.core.HardwareDetector;
import tech.kayys.golek.model.core.HardwareCapabilities;
import tech.kayys.golek.model.ModelRepository;
import tech.kayys.golek.model.ModelManifest;
import tech.kayys.golek.core.exception.ModelNotFoundException;
import tech.kayys.golek.core.exception.NoCompatibleProviderException;

// Additional imports for model classes
import tech.kayys.golek.engine.model.RoutingContext;
import tech.kayys.golek.engine.model.RoutingDecision;
import tech.kayys.golek.engine.model.ProviderCandidate;
import tech.kayys.golek.engine.observability.RuntimeMetricsCache;
import tech.kayys.golek.model.core.SelectionPolicy;

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
    ModelRepository modelRepository;

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
        return Uni.createFrom().item(() -> {

            // Load model manifest
            ModelManifest manifest = modelRepository
                    .findById(modelId, tenantContext.getTenantId())
                    .orElseThrow(() -> new ModelNotFoundException(
                            "Model not found: " + modelId));

            // Build routing context
            RoutingContext context = buildRoutingContext(
                    request,
                    tenantContext,
                    manifest);

            // Select provider
            RoutingDecision decision = selectProvider(manifest, context);

            // Cache decision for debugging
            decisionCache.put(request.getRequestId(), decision);

            LOG.infof("Routing model %s to provider %s (score: %d)",
                    modelId, decision.providerId(), decision.score());

            return decision;
        })
                .onItem().transformToUni(decision -> executeWithProvider(decision, request, tenantContext))
                .onFailure().retry().withBackOff(Duration.ofMillis(100))
                .atMost(3)
                .onFailure().recoverWithUni(error -> handleRoutingFailure(modelId, request, tenantContext, error));
    }

    /**
     * Select best provider using multi-factor scoring
     */
    private RoutingDecision selectProvider(
            ModelManifest manifest,
            RoutingContext context) {
        // Get all available providers
        List<LLMProvider> providers = providerRegistry.all();

        // Filter compatible providers
        List<ProviderCandidate> candidates = providers.stream()
                .filter(p -> p.supports(manifest.modelId(), context.tenantContext()))
                .map(p -> scoreProvider(p, manifest, context))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());

        if (candidates.isEmpty()) {
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

        return new RoutingDecision(
                winner.providerId(),
                winner.provider(),
                winner.score(),
                candidates.stream()
                        .skip(1)
                        .limit(2)
                        .map(c -> c.providerId())
                        .collect(Collectors.toList()),
                manifest,
                context);
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
        Set<String> supportedFormats = provider.capabilities()
                .getSupportedFormats();

        return manifest.artifacts().keySet().stream()
                .anyMatch(format -> supportedFormats.contains(format.toString()));
    }

    private int scoreDeviceCompatibility(
            LLMProvider provider,
            RoutingContext context) {
        HardwareCapabilities hw = hardwareDetector.detect();
        Set<String> supportedDevices = provider.capabilities()
                .getSupportedDevices();

        // Preferred device match
        if (context.deviceHint().isPresent() &&
                supportedDevices.contains(context.deviceHint().get())) {
            return 40;
        }

        // CUDA available and supported
        if (hw.hasCUDA() && supportedDevices.contains("cuda")) {
            return 30;
        }

        // CPU fallback
        if (supportedDevices.contains("cpu")) {
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
        ProviderHealth health = provider.health();

        if (!health.isHealthy()) {
            return -50; // Heavy penalty for unhealthy
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

        // Check memory requirements
        if (manifest.resourceRequirements() != null) {
            long requiredMemory = manifest.resourceRequirements()
                    .minMemory().toBytes();

            if (hw.getAvailableMemory() < requiredMemory) {
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
                context.getTenantId(),
                provider.id());
    }

    private boolean isCircuitBreakerOpen(LLMProvider provider) {
        // Circuit breaker check will be implemented next
        return false;
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

            return providerRegistry.get(fallbackId)
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

        return new RoutingContext(
                request,
                tenantContext,
                request.getPreferredProvider(),
                extractDeviceHint(request),
                timeout,
                isCostSensitive(request, tenantContext),
                request.getPriority());
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
        return tenantConfigRepository.isCostSensitive(context.getTenantId());
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