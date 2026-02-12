/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.tool;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;
import tech.kayys.golek.spi.tool.ToolCall;
import tech.kayys.golek.tool.dto.ToolExecutionResult;
import tech.kayys.golek.tool.util.ToolResultFormatter;
import tech.kayys.golek.tool.validation.ToolArgumentValidator;
import tech.kayys.golek.spi.tool.Tool;
import tech.kayys.golek.spi.registry.ToolRegistry;
import tech.kayys.wayang.tool.dto.ToolExecuteRequest;

import java.util.*;

/**
 * Plugin for executing tool calls detected by the Reasoning plugin.
 * <p>
 * Bound to {@link InferencePhase#POST_PROCESSING}.
 * Wraps {@link DefaultToolExecutor} from wayang-tool-core.
 */
@ApplicationScoped
public class ToolExecutionPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ToolExecutionPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/tool-execution";

    private boolean enabled = true;
    private Map<String, Object> config = new HashMap<>();

    // Note: We'll need to update this once we have the DefaultToolExecutor in the
    // new module
    // For now, keeping the old import for the executor
    @Inject
    tech.kayys.wayang.tool.impl.DefaultToolExecutor toolExecutor;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    ToolRegistry toolRegistry;

    private final ToolArgumentValidator validator = new ToolArgumentValidator();
    private final ToolResultFormatter resultFormatter = new ToolResultFormatter();

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.POST_PROCESSING;
    }

    @Override
    public int order() {
        return 20; // Execute AFTER Reasoning loop detection (order 10)
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        LOG.infof("Initialized %s", PLUGIN_ID);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        // 1. Check for detected tool calls from Reasoning plugin
        List<ToolCall> toolCalls = context.getVariable("detectedToolCalls", List.class)
                .orElse(Collections.emptyList());

        if (toolCalls.isEmpty()) {
            return;
        }

        LOG.debugf("Executing %d tool calls", toolCalls.size());

        // Execute all tools in parallel to support multiple tool calls per turn
        List<ToolExecutionResult> results = new ArrayList<>();

        // For now, execute sequentially but with enhanced error handling
        // In the future, this could be optimized for parallel execution
        for (ToolCall call : toolCalls) {
            try {
                // Get the tool from registry to validate arguments
                Tool tool = toolRegistry.getTool(call.getFunction().getName()).await().indefinitely();

                Map<String, Object> arguments = Collections.emptyMap();
                try {
                    String argsJson = call.getFunction().getArguments();
                    if (argsJson != null && !argsJson.trim().isEmpty()) {
                        arguments = objectMapper.readValue(argsJson, Map.class);
                    }
                } catch (Exception e) {
                    LOG.errorf("Error parsing arguments for tool %s: %s", call.getFunction().getName(), e.getMessage());
                    throw new PluginException("Invalid arguments for tool " + call.getFunction().getName(), e);
                }

                // Validate arguments against the tool's schema
                validator.validate(tool, arguments);

                // Execute the tool using the DefaultToolExecutor
                // We need to convert the SPI ToolCall to the format expected by the executor
                tech.kayys.wayang.tool.dto.ToolExecutionResult wayangResult = toolExecutor.execute(
                        call.getFunction().getName(),
                        arguments,
                        Collections.emptyMap() // context
                ).await().indefinitely();

                // Map Wayang result to Golek result
                tech.kayys.golek.tool.dto.ToolExecutionResult result = new tech.kayys.golek.tool.dto.ToolExecutionResult(
                        call.getId(),
                        call.getFunction().getName(),
                        tech.kayys.golek.tool.dto.InvocationStatus.valueOf(wayangResult.status().name()),
                        wayangResult.output(),
                        wayangResult.errorMessage(),
                        wayangResult.executionTimeMs(),
                        wayangResult.metadata(),
                        wayangResult.status().name().equals("SUCCESS"));

                results.add(result);

                LOG.debugf("Tool %s executed successfully", call.getFunction().getName());

            } catch (Exception e) {
                LOG.errorf("Error executing tool %s: %s", call.getFunction().getName(), e.getMessage());

                // Create an error result to include in the response
                ToolExecutionResult errorResult = ToolExecutionResult.failure(
                        call.getId(),
                        call.getFunction().getName(),
                        "Error: " + e.getMessage(),
                        0 // execution time
                );
                results.add(errorResult);
            }
        }

        // Store raw results
        context.putVariable("toolResults", results);
        context.putVariable("hasToolResults", !results.isEmpty());

        // Format and inject results back into the conversation for the LLM to process
        if (!results.isEmpty()) {
            Map<String, Object> toolResultMessage = resultFormatter.createToolResultMessage(results);
            context.putVariable("toolResultMessage", toolResultMessage);

            // Add to conversation history so the next iteration of the reasoning loop can
            // use it
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> conversationHistory = context.getVariable("conversationHistory", List.class)
                    .orElse(new ArrayList<>());

            conversationHistory.add(toolResultMessage);
            context.putVariable("conversationHistory", conversationHistory);
        }

        // Signal reasoning loop that we have new information
        context.putVariable("reasoningLoopContinue", true);
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of("enabled", enabled);
    }
}
