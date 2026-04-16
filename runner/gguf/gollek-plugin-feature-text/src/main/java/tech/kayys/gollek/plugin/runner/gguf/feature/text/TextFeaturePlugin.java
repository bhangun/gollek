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

package tech.kayys.gollek.plugin.runner.gguf.feature.text;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.inference.gguf.LlamaCppRunner;
import tech.kayys.gollek.inference.gguf.GGUFSession;
import tech.kayys.gollek.plugin.runner.gguf.feature.GGUFFeaturePlugin;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;

import java.util.Map;
import java.util.Set;

/**
 * Text generation feature plugin for GGUF.
 * 
 * <p>Provides text generation capabilities using LlamaCppRunner:
 * <ul>
 *   <li>Text completion</li>
 *   <li>Chat/completion</li>
 *   <li>Code generation</li>
 *   <li>Text embedding</li>
 * </ul>
 * 
 * <h2>Supported Models</h2>
 * <ul>
 *   <li>Llama 2/3</li>
 *   <li>Mistral</li>
 *   <li>Mixtral MoE</li>
 *   <li>Qwen</li>
 *   <li>Falcon</li>
 *   <li>Gemma</li>
 *   <li>Phi</li>
 * </ul>
 * 
 * @since 2.1.0
 */
public class TextFeaturePlugin implements GGUFFeaturePlugin {

    public static final String ID = "text-feature";

    private final LlamaCppRunner llamaCppRunner;
    private boolean enabled = true;
    private String defaultModel = "llama-3-8b";
    private float temperature = 0.7f;
    private int maxTokens = 2048;

    /**
     * Create text feature plugin.
     * 
     * @param llamaCppRunner LlamaCppRunner instance (CDI injected)
     */
    public TextFeaturePlugin(LlamaCppRunner llamaCppRunner) {
        this.llamaCppRunner = llamaCppRunner;
    }

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "Text Generation";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public String description() {
        return "Text generation capabilities using LlamaCpp for LLM inference";
    }

    @Override
    public void initialize(Map<String, Object> config) {
        if (config.containsKey("enabled")) {
            this.enabled = Boolean.parseBoolean(config.get("enabled").toString());
        }
        if (config.containsKey("default_model")) {
            this.defaultModel = config.get("default_model").toString();
        }
        if (config.containsKey("temperature")) {
            this.temperature = Float.parseFloat(config.get("temperature").toString());
        }
        if (config.containsKey("max_tokens")) {
            this.maxTokens = Integer.parseInt(config.get("max_tokens").toString());
        }
    }

    @Override
    public boolean isAvailable() {
        return enabled && llamaCppRunner != null;
    }

    @Override
    public int priority() {
        return 100; // Highest priority for text feature
    }

    @Override
    public Set<String> supportedModels() {
        return Set.of(
            "llama-2-7b", "llama-2-13b", "llama-2-70b",
            "llama-3-8b", "llama-3-70b",
            "mistral-7b", "mixtral-8x7b",
            "qwen-7b", "qwen-14b", "qwen-72b",
            "falcon-7b", "falcon-40b",
            "gemma-7b", "gemma-2b",
            "phi-2", "phi-3"
        );
    }

    @Override
    public Set<String> supportedTasks() {
        return Set.of(
            "completion", "chat", "code_generation",
            "embedding", "summarization", "translation"
        );
    }

    @Override
    public Object process(Object input) {
        if (!isAvailable()) {
            throw new IllegalStateException("Text feature is not available");
        }

        if (input instanceof InferenceRequest) {
            // Standard inference request - return Uni
            return processInferenceRequest((InferenceRequest) input);
        } else if (input instanceof String) {
            // Simple text completion - return Uni
            return processTextCompletion((String) input);
        } else if (input instanceof TextGenerationInput) {
            // Structured text input - return Uni
            return processStructuredInput((TextGenerationInput) input);
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + input.getClass());
        }
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
            "type", "text",
            "runner_available", llamaCppRunner != null,
            "supported_tasks", supportedTasks(),
            "default_model", defaultModel,
            "temperature", temperature,
            "max_tokens", maxTokens,
            "supported_languages", Set.of("en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"),
            "quantization_support", true,
            "lora_support", true
        );
    }

    @Override
    public void shutdown() {
        enabled = false;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private Uni<InferenceResponse> processInferenceRequest(InferenceRequest request) {
        // Use existing LlamaCppRunner implementation
        InferenceResponse response = llamaCppRunner.infer(request);
        return Uni.createFrom().item(response);
    }

    private Uni<String> processTextCompletion(String text) {
        // Create simple inference request for text completion
        InferenceRequest request = InferenceRequest.builder()
            .model(defaultModel)
            .message(Message.user(text))
            .parameter("temperature", temperature)
            .parameter("max_tokens", maxTokens)
            .build();

        return processInferenceRequest(request)
            .map(InferenceResponse::getContent);
    }

    private Object processStructuredInput(TextGenerationInput input) {
        InferenceRequest request = InferenceRequest.builder()
            .model(input.model != null ? input.model : defaultModel)
            .messages(input.messages)
            .parameter("temperature", input.temperature != null ? input.temperature : temperature)
            .parameter("max_tokens", input.maxTokens != null ? input.maxTokens : maxTokens)
            .parameter("top_p", input.topP)
            .parameter("frequency_penalty", input.frequencyPenalty)
            .parameter("presence_penalty", input.presencePenalty)
            .build();

        if ("completion".equals(input.task)) {
            return processInferenceRequest(request);
        } else if ("embedding".equals(input.task)) {
            // TODO: Implement embedding when supported
            return Uni.createFrom().failure(
                new UnsupportedOperationException("Embedding not yet supported")
            );
        } else {
            return processInferenceRequest(request);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Helper Classes
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Structured text generation input.
     */
    public static class TextGenerationInput {
        public final String model;
        public final java.util.List<Message> messages;
        public final String task;
        public final Float temperature;
        public final Integer maxTokens;
        public final Float topP;
        public final Float frequencyPenalty;
        public final Float presencePenalty;

        public TextGenerationInput(
                String model,
                java.util.List<Message> messages,
                String task,
                Float temperature,
                Integer maxTokens,
                Float topP,
                Float frequencyPenalty,
                Float presencePenalty) {
            this.model = model;
            this.messages = messages;
            this.task = task;
            this.temperature = temperature;
            this.maxTokens = maxTokens;
            this.topP = topP;
            this.frequencyPenalty = frequencyPenalty;
            this.presencePenalty = presencePenalty;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private String model;
            private java.util.List<Message> messages;
            private String task = "completion";
            private Float temperature;
            private Integer maxTokens;
            private Float topP;
            private Float frequencyPenalty;
            private Float presencePenalty;

            public Builder model(String model) {
                this.model = model;
                return this;
            }

            public Builder messages(java.util.List<Message> messages) {
                this.messages = messages;
                return this;
            }

            public Builder task(String task) {
                this.task = task;
                return this;
            }

            public Builder temperature(Float temperature) {
                this.temperature = temperature;
                return this;
            }

            public Builder maxTokens(Integer maxTokens) {
                this.maxTokens = maxTokens;
                return this;
            }

            public Builder topP(Float topP) {
                this.topP = topP;
                return this;
            }

            public Builder frequencyPenalty(Float frequencyPenalty) {
                this.frequencyPenalty = frequencyPenalty;
                return this;
            }

            public Builder presencePenalty(Float presencePenalty) {
                this.presencePenalty = presencePenalty;
                return this;
            }

            public TextGenerationInput build() {
                return new TextGenerationInput(
                    model, messages, task, temperature, maxTokens, topP, frequencyPenalty, presencePenalty
                );
            }
        }
    }
}
