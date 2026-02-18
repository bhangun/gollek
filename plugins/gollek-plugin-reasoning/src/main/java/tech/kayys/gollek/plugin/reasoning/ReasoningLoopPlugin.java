/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.reasoning;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.gollek.core.execution.ExecutionContext;
import tech.kayys.gollek.core.plugin.InferencePhasePlugin;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.plugin.PluginException;
import tech.kayys.gollek.spi.tool.ToolCall;

import java.util.*;

/**
 * Plugin implementing multi-turn reasoning loop.
 * <p>
 * Bound to {@link InferencePhase#POST_PROCESSING}.
 * After the LLM produces output, this plugin:
 * <ol>
 *   <li>Parses the output for tool calls</li>
 *   <li>Detects whether to continue the loop or return a final answer</li>
 *   <li>Attempts to repair malformed JSON/tool calls</li>
 *   <li>Tracks reasoning step count and enforces max steps</li>
 * </ol>
 */
@ApplicationScoped
public class ReasoningLoopPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(ReasoningLoopPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/reasoning-loop";

    private boolean enabled = true;
    private int maxSteps = 10;
    private int maxRepairAttempts = 2;
    private Map<String, Object> config = new HashMap<>();

    private final OutputParser outputParser = new OutputParser();
    private final SelfRepairStrategy repairStrategy = new SelfRepairStrategy();

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
        return 10; // Early in POST_PROCESSING to detect tool calls before other post-processors
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("maxSteps").ifPresent(v -> this.maxSteps = Integer.parseInt(v));
        context.getConfig("maxRepairAttempts").ifPresent(v -> this.maxRepairAttempts = Integer.parseInt(v));
        LOG.infof("Initialized %s (maxSteps: %d, maxRepairAttempts: %d)",
                PLUGIN_ID, maxSteps, maxRepairAttempts);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceResponse response = context.getVariable("response", InferenceResponse.class)
                .orElse(null);

        if (response == null) {
            LOG.debug("No response to process for reasoning loop");
            return;
        }

        String output = response.getContent();
        if (output == null || output.isBlank()) {
            return;
        }

        int currentStep = context.getVariable("reasoningStep", Integer.class).orElse(0);

        // 1. Parse for tool calls
        List<ToolCall> toolCalls = outputParser.parseToolCalls(output);

        if (!toolCalls.isEmpty()) {
            LOG.debugf("Detected %d tool call(s) at step %d", toolCalls.size(), currentStep);
            context.putVariable("detectedToolCalls", toolCalls);
            context.putVariable("reasoningContinue", currentStep < maxSteps);
            context.putVariable("reasoningStep", currentStep + 1);
            return;
        }

        // 2. Check for malformed output and attempt repair
        if (outputParser.isMalformed(output)) {
            LOG.debugf("Malformed output detected at step %d, attempting repair", currentStep);
            int repairAttempts = context.getVariable("repairAttempts", Integer.class).orElse(0);

            if (repairAttempts < maxRepairAttempts) {
                Optional<String> repaired = repairStrategy.repair(output);
                if (repaired.isPresent()) {
                    context.putVariable("repairedOutput", repaired.get());
                    context.putVariable("repairAttempts", repairAttempts + 1);
                    context.putVariable("reasoningContinue", true);
                    context.putVariable("wasRepaired", true);
                    LOG.debugf("Output repaired successfully");
                    return;
                }
            }

            LOG.warnf("Could not repair malformed output after %d attempts", repairAttempts);
        }

        // 3. Final answer — extract clean answer
        String finalAnswer = outputParser.extractFinalAnswer(output);
        context.putVariable("finalAnswer", finalAnswer);
        context.putVariable("reasoningContinue", false);
        context.putVariable("totalReasoningSteps", currentStep + 1);

        LOG.debugf("Final answer reached after %d steps", currentStep + 1);
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.maxSteps = (Integer) newConfig.getOrDefault("maxSteps", 10);
        this.maxRepairAttempts = (Integer) newConfig.getOrDefault("maxRepairAttempts", 2);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "maxSteps", maxSteps,
                "maxRepairAttempts", maxRepairAttempts);
    }
}
