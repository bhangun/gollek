/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.engine.registry;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.Message;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Unit tests for RunnerPluginRegistry.
 */
@DisplayName("RunnerPluginRegistry Tests")
class RunnerPluginRegistryTest {

    private RunnerPluginRegistry registry;

    @BeforeEach
    void setUp() {
        registry = new RunnerPluginRegistry();
        // Manually initialize for tests (skip CDI)
        registry.initialized = true;
    }

    @Test
    @DisplayName("Should create registry instance")
    void shouldCreateRegistry() {
        assertNotNull(registry);
        assertFalse(registry.initialized);
    }

    @Test
    @DisplayName("Should discover and register runner plugins")
    void shouldDiscoverRunners() {
        // Create mock runner plugin
        RunnerPlugin mockPlugin = new MockRunnerPlugin("mock-runner", true);

        // Register manually (simulating CDI discovery)
        registry.pluginManager.register(mockPlugin);

        // Verify registration
        List<RunnerPlugin> plugins = registry.getAllPlugins();
        assertEquals(1, plugins.size());
        assertEquals("mock-runner", plugins.get(0).id());
    }

    @Test
    @DisplayName("Should create session for supported model")
    void shouldCreateSessionForSupportedModel() {
        // Register GGUF runner
        RunnerPlugin ggufPlugin = new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf"));
        registry.pluginManager.register(ggufPlugin);

        // Create session
        Optional<RunnerSession> session = registry.createSession("model.gguf", Map.of());

        assertTrue(session.isPresent());
        assertEquals("gguf-runner", session.get().getRunner().id());
    }

    @Test
    @DisplayName("Should return empty for unsupported model")
    void shouldReturnEmptyForUnsupportedModel() {
        // Register GGUF runner only
        RunnerPlugin ggufPlugin = new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf"));
        registry.pluginManager.register(ggufPlugin);

        // Try to create session for ONNX model
        Optional<RunnerSession> session = registry.createSession("model.onnx", Map.of());

        assertFalse(session.isPresent());
    }

    @Test
    @DisplayName("Should find plugin for model")
    void shouldFindPluginForModel() {
        // Register multiple runners
        registry.pluginManager.register(new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf")));
        registry.pluginManager.register(new MockRunnerPlugin("onnx-runner", true, Set.of(".onnx")));

        // Find plugin for GGUF model
        Optional<RunnerPlugin> plugin = registry.findPluginForModel("llama-3-8b.gguf");

        assertTrue(plugin.isPresent());
        assertEquals("gguf-runner", plugin.get().id());
    }

    @Test
    @DisplayName("Should close session")
    void shouldCloseSession() {
        // Register runner and create session
        RunnerPlugin plugin = new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf"));
        registry.pluginManager.register(plugin);

        Optional<RunnerSession> sessionOpt = registry.createSession("model.gguf", Map.of());
        assertTrue(sessionOpt.isPresent());

        String sessionId = sessionOpt.get().getSessionId();

        // Close session
        boolean closed = registry.closeSession(sessionId);

        assertTrue(closed);
        assertFalse(registry.getSession(sessionId).isPresent());
    }

    @Test
    @DisplayName("Should get health status")
    void shouldGetHealthStatus() {
        // Register runners
        registry.pluginManager.register(new MockRunnerPlugin("gguf-runner", true));
        registry.pluginManager.register(new MockRunnerPlugin("onnx-runner", false));

        // Get health status
        Map<String, Boolean> health = registry.getHealthStatus();

        assertEquals(2, health.size());
        assertTrue(health.get("gguf-runner"));
        assertFalse(health.get("onnx-runner"));
    }

    @Test
    @DisplayName("Should get registry stats")
    void shouldGetStats() {
        // Register runner
        registry.pluginManager.register(new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf")));

        // Get stats
        Map<String, Object> stats = registry.getStats();

        assertTrue((Boolean) stats.get("initialized"));
        assertEquals(1, stats.get("total_plugins"));
        assertEquals(1, stats.get("available_plugins"));

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> plugins = (List<Map<String, Object>>) stats.get("plugins");
        assertEquals(1, plugins.size());
        assertEquals("gguf-runner", plugins.get(0).get("id"));
    }

    @Test
    @DisplayName("Should initialize with configuration")
    void shouldInitializeWithConfig() {
        Map<String, Object> config = Map.of(
                "gguf-runner", Map.of("enabled", true, "n_gpu_layers", -1),
                "onnx-runner", Map.of("enabled", false));

        assertDoesNotThrow(() -> registry.initialize(config));
    }

    @Test
    @DisplayName("Should shutdown gracefully")
    void shouldShutdownGracefully() {
        // Register runner
        registry.pluginManager.register(new MockRunnerPlugin("gguf-runner", true));

        // Shutdown
        assertDoesNotThrow(() -> registry.shutdown());

        // Verify shutdown
        assertTrue(registry.getAllPlugins().isEmpty());
    }

    @Test
    @DisplayName("Should handle multiple sessions")
    void shouldHandleMultipleSessions() {
        // Register runner
        RunnerPlugin plugin = new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf"));
        registry.pluginManager.register(plugin);

        // Create multiple sessions
        Optional<RunnerSession> session1 = registry.createSession("model1.gguf", Map.of());
        Optional<RunnerSession> session2 = registry.createSession("model2.gguf", Map.of());
        Optional<RunnerSession> session3 = registry.createSession("model3.gguf", Map.of());

        assertTrue(session1.isPresent());
        assertTrue(session2.isPresent());
        assertTrue(session3.isPresent());

        // Verify unique session IDs
        assertNotEquals(session1.get().getSessionId(), session2.get().getSessionId());
        assertNotEquals(session2.get().getSessionId(), session3.get().getSessionId());
    }

    @Test
    @DisplayName("Should prioritize runners by priority")
    void shouldPrioritizeRunners() {
        // Register runners with different priorities
        registry.pluginManager.register(new MockRunnerPlugin("low-priority", true, Set.of(".gguf"), 10));
        registry.pluginManager.register(new MockRunnerPlugin("high-priority", true, Set.of(".gguf"), 100));
        registry.pluginManager.register(new MockRunnerPlugin("medium-priority", true, Set.of(".gguf"), 50));

        // Get available plugins (should be sorted by priority)
        List<RunnerPlugin> available = registry.getAvailablePlugins();

        assertEquals(3, available.size());
        assertEquals("high-priority", available.get(0).id());
        assertEquals("medium-priority", available.get(1).id());
        assertEquals("low-priority", available.get(2).id());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Mock Runner Plugin for Testing
    // ───────────────────────────────────────────────────────────────────────

    static class MockRunnerPlugin implements RunnerPlugin {
        private final String id;
        private final boolean available;
        private final Set<String> formats;
        private final int priority;

        MockRunnerPlugin(String id, boolean available) {
            this(id, available, Set.of(".gguf"), 0);
        }

        MockRunnerPlugin(String id, boolean available, Set<String> formats) {
            this(id, available, formats, 0);
        }

        MockRunnerPlugin(String id, boolean available, Set<String> formats, int priority) {
            this.id = id;
            this.available = available;
            this.formats = formats;
            this.priority = priority;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public String name() {
            return id;
        }

        @Override
        public String description() {
            return "Mock runner for testing";
        }

        @Override
        public String format() {
            return "mock";
        }

        @Override
        public Set<String> supportedFormats() {
            return formats;
        }

        @Override
        public Set<String> supportedArchitectures() {
            return Set.of("llama", "mistral");
        }

        @Override
        public boolean supportsModel(String modelPath) {
            return formats.stream().anyMatch(f -> modelPath.toLowerCase().endsWith(f));
        }

        @Override
        public boolean isAvailable() {
            return available;
        }

        @Override
        public int priority() {
            return priority;
        }

        @Override
        public RunnerSession createSession(String modelPath, Map<String, Object> config) {
            return new MockRunnerSession(modelPath, this);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Mock Runner Session for Testing
    // ───────────────────────────────────────────────────────────────────────

    static class MockRunnerSession implements RunnerSession {
        private final String sessionId;
        private final String modelPath;
        private final RunnerPlugin runner;
        private final boolean active;

        MockRunnerSession(String modelPath, RunnerPlugin runner) {
            this.sessionId = java.util.UUID.randomUUID().toString();
            this.modelPath = modelPath;
            this.runner = runner;
            this.active = true;
        }

        @Override
        public String getSessionId() {
            return sessionId;
        }

        @Override
        public String getModelPath() {
            return modelPath;
        }

        @Override
        public RunnerPlugin getRunner() {
            return runner;
        }

        @Override
        public Uni<InferenceResponse> infer(InferenceRequest request) {
            return Uni.createFrom().item(
                    InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content("Mock inference response")
                            .model(modelPath)
                            .inputTokens(10)
                            .outputTokens(20)
                            .build());
        }

        @Override
        public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
            return Multi.createFrom().items(
                    StreamingInferenceChunk.of(request.getRequestId(), 0, "Hello "),
                    StreamingInferenceChunk.of(request.getRequestId(), 1, "from "),
                    StreamingInferenceChunk.of(request.getRequestId(), 2, "mock "),
                    StreamingInferenceChunk.of(request.getRequestId(), 3, "runner!"),
                    StreamingInferenceChunk.finalChunk(request.getRequestId(), 4, ""));
        }

        @Override
        public Map<String, Object> getConfig() {
            return Map.of();
        }

        @Override
        public boolean isActive() {
            return active;
        }

        @Override
        public void close() {
            // Mock close
        }

        @Override
        public ModelInfo getModelInfo() {
            return new ModelInfo(modelPath, "llama", 7_000_000_000L, 4096, 4096, Map.of());
        }
    }
}
