/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.sampling;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;

import java.util.*;

/**
 * Plugin for sampling and decoding parameter policy control.
 * <p>
 * Bound to {@link InferencePhase#PRE_PROCESSING}.
 * Normalizes, validates, and applies default/constraint sampling parameters:
 * temperature, top-k, top-p, repetition penalty, presence penalty, stop tokens, grammar/JSON mode.
 * <p>
 * Golek = policy, llama.cpp/LiteRT = execution.
 */
@ApplicationScoped
public class SamplingPolicyPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(SamplingPolicyPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/sampling-policy";

    private boolean enabled = true;
    private Map<String, Object> config = new HashMap<>();

    // Default sampling parameters
    private double defaultTemperature = 0.7;
    private int defaultTopK = 40;
    private double defaultTopP = 0.95;
    private double defaultRepetitionPenalty = 1.1;
    private double defaultPresencePenalty = 0.0;
    private int defaultMaxTokens = 2048;

    // Constraints
    private double maxTemperature = 2.0;
    private double minTemperature = 0.0;
    private int maxMaxTokens = 8192;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return 5; // Very early in PRE_PROCESSING, before prompt building
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("defaultTemperature").ifPresent(v -> this.defaultTemperature = Double.parseDouble(v));
        context.getConfig("defaultTopK").ifPresent(v -> this.defaultTopK = Integer.parseInt(v));
        context.getConfig("defaultTopP").ifPresent(v -> this.defaultTopP = Double.parseDouble(v));
        context.getConfig("defaultRepetitionPenalty").ifPresent(v -> this.defaultRepetitionPenalty = Double.parseDouble(v));
        context.getConfig("defaultPresencePenalty").ifPresent(v -> this.defaultPresencePenalty = Double.parseDouble(v));
        context.getConfig("defaultMaxTokens").ifPresent(v -> this.defaultMaxTokens = Integer.parseInt(v));
        context.getConfig("maxTemperature").ifPresent(v -> this.maxTemperature = Double.parseDouble(v));
        context.getConfig("maxMaxTokens").ifPresent(v -> this.maxMaxTokens = Integer.parseInt(v));
        LOG.infof("Initialized %s (defaults: temp=%.1f, topK=%d, topP=%.2f)",
                PLUGIN_ID, defaultTemperature, defaultTopK, defaultTopP);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new PluginException("Request not found in execution context"));

        // Build normalized sampling config
        SamplingConfig samplingConfig = buildSamplingConfig(request);

        // Validate constraints
        validateConstraints(samplingConfig);

        // Store in context for provider dispatch
        context.putVariable("samplingConfig", samplingConfig);

        LOG.debugf("Applied sampling policy: %s", samplingConfig);
    }

    /**
     * Build sampling config from request, applying defaults for missing values.
     */
    SamplingConfig buildSamplingConfig(InferenceRequest request) {
        Map<String, Object> params = request.getParameters() != null
                ? request.getParameters() : Collections.emptyMap();

        return new SamplingConfig(
                getDouble(params, "temperature", defaultTemperature),
                getInt(params, "top_k", defaultTopK),
                getDouble(params, "top_p", defaultTopP),
                getDouble(params, "repetition_penalty", defaultRepetitionPenalty),
                getDouble(params, "presence_penalty", defaultPresencePenalty),
                getInt(params, "max_tokens", defaultMaxTokens),
                getStringList(params, "stop"),
                getString(params, "grammar_mode", null)
        );
    }

    /**
     * Validate sampling parameters against configured constraints.
     */
    void validateConstraints(SamplingConfig samplingConfig) throws PluginException {
        if (samplingConfig.temperature() < minTemperature || samplingConfig.temperature() > maxTemperature) {
            throw new PhasePluginException(
                    String.format("Temperature %.2f out of range [%.2f, %.2f]",
                            samplingConfig.temperature(), minTemperature, maxTemperature));
        }

        if (samplingConfig.maxTokens() > maxMaxTokens) {
            throw new PhasePluginException(
                    String.format("maxTokens %d exceeds maximum %d",
                            samplingConfig.maxTokens(), maxMaxTokens));
        }

        if (samplingConfig.topP() < 0.0 || samplingConfig.topP() > 1.0) {
            throw new PhasePluginException(
                    String.format("top_p %.2f out of range [0.0, 1.0]", samplingConfig.topP()));
        }

        if (samplingConfig.topK() < 0) {
            throw new PhasePluginException(
                    String.format("top_k %d must be non-negative", samplingConfig.topK()));
        }
    }

    private double getDouble(Map<String, Object> params, String key, double defaultValue) {
        Object val = params.get(key);
        if (val instanceof Number n) return n.doubleValue();
        return defaultValue;
    }

    private int getInt(Map<String, Object> params, String key, int defaultValue) {
        Object val = params.get(key);
        if (val instanceof Number n) return n.intValue();
        return defaultValue;
    }

    private String getString(Map<String, Object> params, String key, String defaultValue) {
        Object val = params.get(key);
        if (val instanceof String s) return s;
        return defaultValue;
    }

    @SuppressWarnings("unchecked")
    private List<String> getStringList(Map<String, Object> params, String key) {
        Object val = params.get(key);
        if (val instanceof List<?> list) {
            return list.stream().map(Object::toString).toList();
        }
        return List.of();
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.defaultTemperature = ((Number) newConfig.getOrDefault("defaultTemperature", 0.7)).doubleValue();
        this.defaultTopK = ((Number) newConfig.getOrDefault("defaultTopK", 40)).intValue();
        this.defaultTopP = ((Number) newConfig.getOrDefault("defaultTopP", 0.95)).doubleValue();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "defaultTemperature", defaultTemperature,
                "defaultTopK", defaultTopK,
                "defaultTopP", defaultTopP,
                "defaultMaxTokens", defaultMaxTokens);
    }
}
