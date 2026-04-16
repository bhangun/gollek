package tech.kayys.gollek.plugin.cache;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.Optional;

/**
 * Semantic cache plugin that intercepts inference requests to check cache
 * before forwarding to models, and caches responses after inference.
 *
 * <h3>Phases:</h3>
 * <ul>
 * <li>{@code PRE_PROCESSING} - Check cache for similar requests</li>
 * <li>{@code POST_PROCESSING} - Cache successful responses</li>
 * </ul>
 *
 * <h3>Configuration:</h3>
 * <ul>
 * <li>{@code gollek.semantic-cache.enabled} - Enable/disable caching (default: true)</li>
 * <li>{@code gollek.semantic-cache.threshold} - Similarity threshold 0.0-1.0 (default: 0.85)</li>
 * <li>{@code gollek.semantic-cache.max-size} - Maximum cache entries (default: 10000)</li>
 * <li>{@code gollek.semantic-cache.ttl} - Cache TTL in hours (default: 24)</li>
 * </ul>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class SemanticCacheLookupPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(SemanticCacheLookupPlugin.class);

    private static final String PLUGIN_ID = "semantic-cache-lookup";
    private static final String VERSION = "1.0.0";

    @Inject
    SemanticCacheService cacheService;

    private boolean enabled = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        // Execute very early in PRE_PROCESSING to return cached results
        return 0;
    }

    @Override
    public void initialize(tech.kayys.gollek.spi.plugin.PluginContext context) {
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));
        LOG.infof("Initialized %s plugin (enabled: %b)", PLUGIN_ID, enabled);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            return false;
        }

        Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
        if (requestOpt.isEmpty()) {
            return false;
        }

        InferenceRequest request = requestOpt.get();

        // Don't cache streaming requests
        if (request.isStreaming()) {
            return false;
        }

        // Don't cache if explicitly disabled in request
        if (Boolean.TRUE.equals(request.getMetadata().get("no_cache"))) {
            return false;
        }

        return true;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        LOG.debug("Checking semantic cache for request");

        try {
            Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
            if (requestOpt.isPresent()) {
                InferenceRequest request = requestOpt.get();
                Optional<InferenceResponse> cachedResponse = cacheService.get(request);

                if (cachedResponse.isPresent()) {
                    LOG.info("Semantic cache HIT");

                    InferenceResponse response = cachedResponse.get();
                    
                    // Mark as cache hit in metadata
                    InferenceResponse hitResponse = response.toBuilder()
                        .metadata("cache_hit", true)
                        .metadata("cache_type", "semantic")
                        .metadata("cache_timestamp", System.currentTimeMillis())
                        .build();

                    // Store in context to return immediately
                    context.putVariable("response", hitResponse);
                    
                    LOG.infof("Returned cached response for request %s", request.getRequestId());
                }
            }
        } catch (Exception e) {
            LOG.warnf(e, "Cache check failed, proceeding without cache");
        }
    }
}
