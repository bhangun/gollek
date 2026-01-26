package tech.kayys.golek.provider.core.mcp;

import io.smallrye.mutiny.Uni;
import io.vertx.core.spi.launcher.ExecutionContext;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.provider.core.inference.InferencePhasePlugin;
import tech.kayys.golek.provider.core.plugin.ConfigurablePlugin;
import tech.kayys.golek.provider.core.plugin.Plugin.PluginMetadata;

import org.jboss.logging.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * Plugin that adds MCP tool execution to the inference pipeline.
 * Executes during PRE_PROCESSING phase to resolve tool calls.
 */
@ApplicationScoped
public class MCPToolPlugin implements InferencePhasePlugin, ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(MCPToolPlugin.class);
    private static final String PLUGIN_ID = "mcp-tools";

    @Inject
    MCPToolExecutor toolExecutor;

    @Inject
    MCPToolRegistry toolRegistry;

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "MCP Tool Executor";
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
        return 50; // Execute before other pre-processing plugins
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        this.config = new HashMap<>(context.config());
        this.enabled = context.getConfigOrDefault("enabled", true);

        LOG.infof("Initialized MCP Tool Plugin (enabled: %s)", enabled);
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            return false;
        }

        // Check if request contains tool calls
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) context.getVariable("request_parameters", Map.class)
                .orElse(Map.of());

        return params.containsKey("tools");
    }

    @Override
    public Uni<Void> execute(ExecutionContext context) {
        LOG.debug("Executing MCP tool plugin");

        // Extract tool calls from context
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) context.getVariable("request_parameters", Map.class)
                .orElse(Map.of());

        @SuppressWarnings("unchecked")
        Map<String, Map<String, Object>> toolCalls = (Map<String, Map<String, Object>>) params.get("tools");

        if (toolCalls == null || toolCalls.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        // Execute tools
        return toolExecutor.executeTools(toolCalls)
                .onItem().invoke(results -> {
                    // Store results in context
                    context.putVariable("mcp_tool_results", results);
                    context.putMetadata("mcp_tools_executed", results.size());

                    LOG.debugf("Executed %d MCP tools", results.size());
                })
                .replaceWithVoid();
    }

    @Override
    public boolean onFailure(ExecutionContext context, Throwable error) {
        LOG.errorf(error, "MCP tool execution failed");

        // Store error but allow pipeline to continue
        context.putVariable("mcp_tool_error", error.getMessage());
        context.putMetadata("mcp_tool_failed", true);

        return true; // Continue pipeline
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
        LOG.info("Updating MCP tool plugin configuration");
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean validateConfig(Map<String, Object> config) {
        // Validate configuration structure
        return config != null;
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    @Override
    public PluginHealth health() {
        try {
            int availableTools = toolRegistry.listAllTools().size();

            if (availableTools == 0) {
                return PluginHealth.degraded(
                        "No MCP tools available",
                        Map.of("toolCount", 0));
            }

            return PluginHealth.healthy(
                    String.format("%d MCP tools available", availableTools));
        } catch (Exception e) {
            return PluginHealth.unhealthy(
                    "Failed to check tool availability: " + e.getMessage());
        }
    }

    @Override
    public PluginMetadata metadata() {
        return PluginMetadata.builder()
                .id(id())
                .name(name())
                .version(version())
                .description("Executes MCP tools during inference pre-processing")
                .author("Kayys Tech")
                .tag("mcp")
                .tag("tools")
                .tag("pre-processing")
                .property("phase", phase().name())
                .property("order", String.valueOf(order()))
                .build();
    }
}