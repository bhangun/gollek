package tech.kayys.golek.provider.core.routing;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.ModelProviderMapping;
import tech.kayys.golek.api.routing.ProviderPool;
import tech.kayys.golek.api.routing.QuotaExhaustedException;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.api.routing.RoutingDecision;
import tech.kayys.golek.api.routing.SelectionStrategy;
import tech.kayys.golek.provider.core.registry.ProviderRegistry;
import tech.kayys.golek.provider.core.routing.strategy.*;
import tech.kayys.golek.provider.core.spi.LLMProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Multi-provider router with pluggable selection strategies.
 * Handles provider selection, load balancing, and automatic failover.
 * Routes requests based on model-provider mappings.
 */
@ApplicationScoped
public class MultiProviderRouter {

    private static final Logger LOG = Logger.getLogger(MultiProviderRouter.class);

    @Inject
    ProviderRegistry providerRegistry;

    @Inject
    ModelProviderRegistry modelProviderRegistry;

    private final Map<SelectionStrategy, ProviderSelector> strategies = new ConcurrentHashMap<>();
    private RoutingConfig config = RoutingConfig.defaults();

    public MultiProviderRouter() {
        initializeStrategies();
    }

    private void initializeStrategies() {
        strategies.put(SelectionStrategy.ROUND_ROBIN, new RoundRobinSelector());
        strategies.put(SelectionStrategy.RANDOM, new RandomSelector());
        strategies.put(SelectionStrategy.WEIGHTED_RANDOM, new WeightedRandomSelector());
        strategies.put(SelectionStrategy.LEAST_LOADED, new LeastLoadedSelector());
        strategies.put(SelectionStrategy.COST_OPTIMIZED, new CostOptimizedSelector());
        strategies.put(SelectionStrategy.LATENCY_OPTIMIZED, new LatencyOptimizedSelector());
        strategies.put(SelectionStrategy.FAILOVER, new FailoverSelector());
        strategies.put(SelectionStrategy.SCORED, new ScoredSelector());
        strategies.put(SelectionStrategy.USER_SELECTED, new UserSelectedSelector());
    }

    /**
     * Configure the router
     */
    public void configure(RoutingConfig config) {
        this.config = config;
        LOG.infof("Router configured with default strategy: %s, pools: %d",
                config.defaultStrategy(), config.pools().size());
    }

    /**
     * Select optimal provider for a model and routing context.
     * Uses model-provider mappings to determine eligible providers.
     */
    public Uni<RoutingDecision> selectProvider(
            String modelId,
            RoutingContext context) {

        return Uni.createFrom().item(() -> {
            // Get effective strategy
            SelectionStrategy strategy = context.getEffectiveStrategy(config.defaultStrategy());
            LOG.debugf("Selecting provider for model %s using strategy %s", modelId, strategy);

            // Get candidate providers based on model mapping
            List<LLMProvider> candidates = getCandidates(modelId, context);
            if (candidates.isEmpty()) {
                throw new IllegalStateException("No available providers for model: " + modelId);
            }

            // Apply selection strategy
            ProviderSelector selector = strategies.get(strategy);
            if (selector == null) {
                LOG.warnf("No selector for strategy %s, falling back to ROUND_ROBIN", strategy);
                selector = strategies.get(SelectionStrategy.ROUND_ROBIN);
            }

            // Select provider
            ProviderSelection selection = selector.select(candidates, context, config);

            // Build routing decision
            return RoutingDecision.builder()
                    .selectedProviderId(selection.provider().id())
                    .poolId(selection.poolId())
                    .strategyUsed(strategy)
                    .score(selection.score())
                    .fallbackProviders(selection.fallbacks().stream()
                            .map(LLMProvider::id)
                            .collect(Collectors.toList()))
                    .build();
        });
    }

    /**
     * Select provider with automatic failover on quota exhaustion
     */
    public Uni<LLMProvider> selectWithFailover(
            String modelId,
            RoutingContext context) {

        return selectProvider(modelId, context)
                .map(decision -> providerRegistry.getProviderOrThrow(decision.selectedProviderId()))
                .onFailure(QuotaExhaustedException.class).recoverWithUni(ex -> {
                    QuotaExhaustedException qe = (QuotaExhaustedException) ex;
                    LOG.warnf("Quota exhausted for provider %s, attempting failover", qe.getProviderId());

                    // Exclude exhausted provider and retry
                    RoutingContext updatedContext = context.excludeProvider(qe.getProviderId());
                    return selectWithFailover(modelId, updatedContext);
                });
    }

    /**
     * Get candidate providers for a model, filtered by context.
     * Uses model-provider registry to find which providers support the model.
     */
    public List<LLMProvider> getCandidates(String modelId, RoutingContext context) {
        List<LLMProvider> providers = new ArrayList<>();

        // First, check model-provider registry for explicit mappings
        Optional<ModelProviderMapping> mapping = modelProviderRegistry.getMapping(modelId);

        if (mapping.isPresent()) {
            // Use mapped providers for this model
            List<String> mappedProviderIds = mapping.get().providerIds();
            LOG.debugf("Model %s mapped to providers: %s", modelId, mappedProviderIds);

            for (String providerId : mappedProviderIds) {
                providerRegistry.getProvider(providerId).ifPresent(providers::add);
            }
        } else {
            // Fallback: query all providers that support the model
            LOG.debugf("No explicit mapping for model %s, querying all providers", modelId);
            providers = providerRegistry.getProvidersForModel(modelId);
        }

        // Filter by pool if specified
        if (context.poolId().isPresent()) {
            ProviderPool pool = config.getPool(context.poolId().get());
            if (pool != null) {
                providers = providers.stream()
                        .filter(p -> pool.containsProvider(p.id()))
                        .collect(Collectors.toList());
            }
        }

        // Filter out excluded providers
        providers = providers.stream()
                .filter(p -> !context.isExcluded(p.id()))
                .collect(Collectors.toList());

        // Filter by health
        providers = providers.stream()
                .filter(p -> p.health().isHealthy())
                .collect(Collectors.toList());

        LOG.debugf("Found %d candidate providers for model %s", providers.size(), modelId);
        return providers;
    }

    /**
     * Get model-provider mapping
     */
    public Optional<ModelProviderMapping> getModelMapping(String modelId) {
        return modelProviderRegistry.getMapping(modelId);
    }

    /**
     * Register a model-provider mapping
     */
    public void registerModelMapping(ModelProviderMapping mapping) {
        modelProviderRegistry.register(mapping);
    }

    /**
     * Get all models available on a provider
     */
    public java.util.Set<String> getModelsForProvider(String providerId) {
        return modelProviderRegistry.getModelsForProvider(providerId);
    }

    /**
     * Get pool for a provider
     */
    public Optional<ProviderPool> getPoolForProvider(String providerId) {
        return config.getPoolsForProvider(providerId).stream().findFirst();
    }

    /**
     * Get all configured pools
     */
    public List<ProviderPool> getPools() {
        return config.pools();
    }

    /**
     * Register a custom selection strategy
     */
    public void registerStrategy(SelectionStrategy strategy, ProviderSelector selector) {
        strategies.put(strategy, selector);
        LOG.infof("Registered custom selector for strategy: %s", strategy);
    }

    /**
     * Get current configuration
     */
    public RoutingConfig getConfig() {
        return config;
    }
}
