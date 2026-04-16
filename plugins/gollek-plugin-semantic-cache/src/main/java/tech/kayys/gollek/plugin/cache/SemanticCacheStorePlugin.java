package tech.kayys.gollek.plugin.cache;

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
 * Semantic cache plugin that stores successful inference responses 
 * during the POST_PROCESSING phase.
 */
@ApplicationScoped
public class SemanticCacheStorePlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(SemanticCacheStorePlugin.class);

    private static final String PLUGIN_ID = "semantic-cache-store";
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
        return InferencePhase.POST_PROCESSING;
    }

    @Override
    public int order() {
        // Execute in POST_PROCESSING
        return 50;
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

        // Only store if we have both request and response
        Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
        Optional<InferenceResponse> responseOpt = context.getVariable("response", InferenceResponse.class);

        if (requestOpt.isEmpty() || responseOpt.isEmpty()) {
            return false;
        }

        InferenceRequest request = requestOpt.get();
        InferenceResponse response = responseOpt.get();

        // Don't cache if it was already a cache hit
        if (Boolean.TRUE.equals(response.getMetadata().get("cache_hit"))) {
            return false;
        }

        // Don't cache streaming requests/responses
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
        try {
            InferenceRequest request = context.getVariable("request", InferenceRequest.class).get();
            InferenceResponse response = context.getVariable("response", InferenceResponse.class).get();

            LOG.debugf("Caching inference response for request %s", request.getRequestId());
            cacheService.put(request, response);
            
        } catch (Exception e) {
            LOG.warnf(e, "Failed to cache response");
        }
    }
}
