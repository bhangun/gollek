package tech.kayys.golek.provider.core.mcp;

import io.smallrye.mutiny.Uni;
import io.vertx.core.spi.launcher.ExecutionContext;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.provider.core.inference.InferencePhasePlugin;
import tech.kayys.golek.provider.core.plugin.ConfigurablePlugin;
import tech.kayys.golek.provider.core.plugin.Plugin.PluginMetadata;

import org.jboss.logging.Logger;

import java.util.*;

/**
 * Plugin that fetches MCP resources during inference.
 * Phase-bound to PRE_PROCESSING (executes before tools).
 */
@ApplicationScoped
public class MCPResourcePlugin implements InferencePhasePlugin, ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(MCPResourcePlugin.class);
    private static final String PLUGIN_ID = "mcp-resources";

    @Inject
    MCPResourceProvider resourceProvider;

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "MCP Resource Fetcher";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return 40; // Execute before tools
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        this.config = new HashMap<>(context.config());
        this.enabled = context.getConfigOrDefault("enabled", true);

        LOG.infof("Initialized %s (enabled: %s)", name(), enabled);
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            return false;
        }

        // Check if request has resource URIs
        return context.getVariable("resource_uris", List.class).isPresent();
    }

    @Override
    public Uni<Void> execute(ExecutionContext context) {
        @SuppressWarnings("unchecked")
        List<String> resourceUris = (List<String>) context.getVariable("resource_uris", List.class)
                .orElse(List.of());

        if (resourceUris.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        LOG.debugf("Fetching %d MCP resources", resourceUris.size());

        return resourceProvider.readResources(resourceUris)
                .onItem().invoke(resources -> {
                    context.putVariable("mcp_resources", resources);
                    context.putMetadata("mcp_resources_count", resources.size());
                })
                .replaceWithVoid();
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        return Uni.createFrom().voidItem();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    @Override
    public PluginMetadata metadata() {
        return PluginMetadata.builder()
                .id(id())
                .name(name())
                .version(version())
                .description("Fetches MCP resources during inference")
                .author("Kayys Tech")
                .tag("mcp")
                .tag("resources")
                .dependency(PluginMetadata.PluginDependency.required("mcp-connections", "1.0.0"))
                .build();
    }
}