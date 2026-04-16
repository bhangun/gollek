/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.runner.safetensor.feature.text;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.safetensor.feature.SafetensorFeaturePlugin;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;

/**
 * Text processing feature plugin for Safetensor.
 * 
 * <p>Provides text processing capabilities utilizing the DirectInferenceEngine:
 * <ul>
 *   <li>Text generation (LLMs)</li>
 *   <li>Text embedding (via core engine)</li>
 * </ul>
 */
@ApplicationScoped
public class TextFeaturePlugin implements SafetensorFeaturePlugin {

    private static final Logger LOG = Logger.getLogger(TextFeaturePlugin.class);
    public static final String ID = "text-feature";

    @Inject
    DirectInferenceEngine engine;

    private boolean enabled = true;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "Safetensor Text Feature";
    }

    @Override
    public String version() {
        return "2.0.0";
    }

    @Override
    public String description() {
        return "High-performance text generation and processing for Safetensor models";
    }

    @Override
    public void initialize(Map<String, Object> config) {
        if (config.containsKey("enabled")) {
            this.enabled = Boolean.parseBoolean(config.get("enabled").toString());
        }
        LOG.infof("TextFeaturePlugin initialized (enabled=%s)", enabled);
    }

    @Override
    public boolean isAvailable() {
        return enabled && engine != null;
    }

    @Override
    public int priority() {
        return 100; // High priority for text features
    }

    @Override
    public Set<String> supportedModels() {
        return Set.of(
            "llama", "mistral", "mixtral", "qwen", "phi", "gemma", "falcon", "bert"
        );
    }

    @Override
    public Set<String> supportedInputTypes() {
        return Set.of("text/plain", "application/json");
    }

    @Override
    public Set<String> supportedOutputTypes() {
        return Set.of("text/plain", "application/json");
    }

    @Override
    @SuppressWarnings("unchecked")
    public Object process(Object input) {
        if (!isAvailable()) {
            throw new IllegalStateException("Text feature is not available");
        }

        if (input instanceof String prompt) {
            return processText(prompt, Map.of());
        } else if (input instanceof Map) {
            Map<String, Object> params = (Map<String, Object>) input;
            String prompt = params.getOrDefault("prompt", "").toString();
            return processText(prompt, params);
        }

        throw new IllegalArgumentException("Unsupported input type: " + input.getClass().getName());
    }

    private Object processText(String prompt, Map<String, Object> params) {
        String modelPath = params.getOrDefault("model_path", "").toString();
        if (modelPath.isBlank()) {
            return Map.of("error", "model_path is required for text processing");
        }

        LOG.debugf("Processing text request for model: %s", modelPath);
        
        GenerationConfig config = GenerationConfig.builder()
                .maxNewTokens(extractInt(params, "max_tokens", 512))
                .temperature(extractFloat(params, "temperature", 0.7f))
                .topP(extractFloat(params, "top_p", 0.9f))
                .build();

        return engine.generate(prompt, Paths.get(modelPath), config)
                .await().indefinitely();
    }

    private int extractInt(Map<String, Object> params, String key, int defaultValue) {
        Object val = params.get(key);
        if (val instanceof Number n) return n.intValue();
        return defaultValue;
    }

    private float extractFloat(Map<String, Object> params, String key, float defaultValue) {
        Object val = params.get(key);
        if (val instanceof Number n) return n.floatValue();
        return defaultValue;
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
            "type", "text",
            "capabilities", Set.of("generation", "completion", "chat"),
            "backend", "direct-safetensor"
        );
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down TextFeaturePlugin");
        enabled = false;
    }

    @Override
    public boolean isHealthy() {
        return isAvailable();
    }
}
