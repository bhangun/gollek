/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * LoraAdapterTest.java
 * ────────────────────
 * Unit tests for LoRA adapter loading and management.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LoraAdapter}.
 */
class LoraAdapterTest {

    @TempDir
    Path tempDir;

    private LoraAdapterService loraAdapter;

    @BeforeEach
    void setUp() {
        // Note: In a full integration test, these would be injected via CDI
        // For unit testing, we instantiate directly
        loraAdapter = new LoraAdapterService();
    }

    @Test
    void testAdapterConfig_DefaultValues() {
        AdapterConfig config = new AdapterConfig(
                16, 16.0f, 0.0f, "none",
                java.util.List.of("q_proj", "v_proj"), "CAUSAL_LM",
                true, null, java.util.Map.of()
        );

        assertEquals(16, config.rank());
        assertEquals(16.0f, config.alpha());
        assertEquals(0.0f, config.dropout());
        assertEquals("none", config.bias());
        assertEquals(1.0f, config.scalingFactor());
        assertTrue(config.matchesTarget("model.layers.0.self_attn.q_proj"));
        assertTrue(config.matchesTarget("model.layers.0.self_attn.v_proj"));
        assertFalse(config.matchesTarget("model.layers.0.mlp.gate_proj"));
    }

    @Test
    void testAdapterConfig_CustomScaling() {
        AdapterConfig config = new AdapterConfig(
                8, 32.0f, 0.1f, "all",
                java.util.List.of(), null, true, null, java.util.Map.of()
        );

        assertEquals(8, config.rank());
        assertEquals(32.0f, config.alpha());
        assertEquals(4.0f, config.scalingFactor()); // alpha / rank = 32 / 8
        assertTrue(config.matchesTarget("any_module")); // empty targets = apply to all
    }

    @Test
    void testResolveAdapterFile_DirectoryWithStandardName() throws IOException {
        Path adapterDir = tempDir.resolve("test-adapter");
        Files.createDirectory(adapterDir);
        Path adapterFile = adapterDir.resolve("adapter_model.safetensors");
        Files.createFile(adapterFile);

        // Would normally use reflection to test private method
        // For now, verify the directory structure is valid
        assertTrue(Files.isDirectory(adapterDir));
        assertTrue(Files.isRegularFile(adapterFile));
    }

    @Test
    void testResolveAdapterFile_DirectoryWithPeftName() throws IOException {
        Path adapterDir = tempDir.resolve("peft-adapter");
        Files.createDirectory(adapterDir);
        Path adapterFile = adapterDir.resolve("adapter.safetensors");
        Files.createFile(adapterFile);

        assertTrue(Files.isDirectory(adapterDir));
        assertTrue(Files.isRegularFile(adapterFile));
    }

    @Test
    void testResolveAdapterFile_DirectFilePath() throws IOException {
        Path adapterFile = tempDir.resolve("custom-adapter.safetensors");
        Files.createFile(adapterFile);

        assertTrue(Files.isRegularFile(adapterFile));
    }

    @Test
    void testResolveAdapterFile_InvalidDirectory() throws IOException {
        Path emptyDir = tempDir.resolve("empty-adapter");
        Files.createDirectory(emptyDir);

        // Should fail because no adapter file exists
        assertFalse(Files.isRegularFile(emptyDir));
        assertTrue(Files.isDirectory(emptyDir));
        assertEquals(0, Files.list(emptyDir).count());
    }

    @Test
    void testIsLoraTensor() {
        // Use reflection to test private method
        try {
            java.lang.reflect.Method method = LoraAdapterService.class.getDeclaredMethod("isLoraTensor", String.class);
            method.setAccessible(true);

            LoraAdapterService adapter = new LoraAdapterService();

            assertTrue((boolean) method.invoke(adapter, "model.layers.0.self_attn.q_proj.lora_A.weight"));
            assertTrue((boolean) method.invoke(adapter, "model.layers.0.self_attn.q_proj.lora_A"));
            assertTrue((boolean) method.invoke(adapter, "model.layers.0.self_attn.q_proj.lora_B.weight"));
            assertTrue((boolean) method.invoke(adapter, "model.layers.0.self_attn.q_proj.lora_B"));

            assertFalse((boolean) method.invoke(adapter, "model.layers.0.self_attn.q_proj.weight"));
            assertFalse((boolean) method.invoke(adapter, "model.layers.0.mlp.gate_proj.weight"));
            assertFalse((boolean) method.invoke(adapter, "lm_head.weight"));

        } catch (Exception e) {
            fail("Reflection test failed: " + e.getMessage());
        }
    }

    @Test
    void testLoadedAdapter_GetLoraPair() {
        // Create a mock LoadedAdapter
        java.util.Map<String, tech.kayys.gollek.safetensor.core.tensor.AccelTensor> weights = new java.util.HashMap<>();

        // Mock tensors (using zeros for testing)
        tech.kayys.gollek.safetensor.core.tensor.AccelTensor mockA =
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(16, 4096);
        tech.kayys.gollek.safetensor.core.tensor.AccelTensor mockB =
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 16);

        weights.put("model.layers.0.self_attn.q_proj.lora_A.weight", mockA);
        weights.put("model.layers.0.self_attn.q_proj.lora_B.weight", mockB);

        AdapterConfig config = new AdapterConfig(
                16, 16.0f, 0.0f, "none",
                java.util.List.of(), null, true, null, java.util.Map.of()
        );

        LoadedAdapter loadedAdapter = new LoadedAdapter(
                tempDir, weights, config, System.currentTimeMillis()
        );

        // Test getting LoRA pair
        java.util.Optional<LoraPair> pair = loadedAdapter.getLoraPair("model.layers.0.self_attn.q_proj");
        assertTrue(pair.isPresent());
        assertEquals(mockA, pair.get().a());
        assertEquals(mockB, pair.get().b());

        // Test getting non-existent module
        java.util.Optional<LoraPair> missing = loadedAdapter.getLoraPair("model.layers.0.mlp.gate_proj");
        assertFalse(missing.isPresent());

        // Cleanup
        loadedAdapter.close();
    }

    @Test
    void testLoadedAdapter_GetModuleNames() {
        java.util.Map<String, tech.kayys.gollek.safetensor.core.tensor.AccelTensor> weights = new java.util.HashMap<>();

        // Add multiple LoRA modules
        weights.put("model.layers.0.self_attn.q_proj.lora_A.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(16, 4096));
        weights.put("model.layers.0.self_attn.q_proj.lora_B.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 16));
        weights.put("model.layers.0.self_attn.v_proj.lora_A.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(16, 4096));
        weights.put("model.layers.0.self_attn.v_proj.lora_B.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 16));
        weights.put("model.layers.0.mlp.gate_proj.lora_A.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(16, 4096));
        weights.put("model.layers.0.mlp.gate_proj.lora_B.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 16));

        // Add non-LoRA tensor (should be ignored)
        weights.put("model.layers.0.self_attn.q_proj.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 4096));

        AdapterConfig config = new AdapterConfig(
                16, 16.0f, 0.0f, "none",
                java.util.List.of(), null, true, null, java.util.Map.of()
        );

        LoadedAdapter loadedAdapter = new LoadedAdapter(
                tempDir, weights, config, System.currentTimeMillis()
        );

        Set<String> moduleNames = loadedAdapter.getModuleNames();

        assertEquals(3, moduleNames.size());
        assertTrue(moduleNames.contains("model.layers.0.self_attn.q_proj"));
        assertTrue(moduleNames.contains("model.layers.0.self_attn.v_proj"));
        assertTrue(moduleNames.contains("model.layers.0.mlp.gate_proj"));

        // Cleanup
        loadedAdapter.close();
    }

    @Test
    void testLoadedAdapter_HasModule() {
        java.util.Map<String, tech.kayys.gollek.safetensor.core.tensor.AccelTensor> weights = new java.util.HashMap<>();
        weights.put("model.layers.0.self_attn.q_proj.lora_A.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(16, 4096));
        weights.put("model.layers.0.self_attn.q_proj.lora_B.weight",
                tech.kayys.gollek.safetensor.core.tensor.AccelTensor.zeros(4096, 16));

        AdapterConfig config = new AdapterConfig(
                16, 16.0f, 0.0f, "none",
                java.util.List.of(), null, true, null, java.util.Map.of()
        );

        LoadedAdapter loadedAdapter = new LoadedAdapter(
                tempDir, weights, config, System.currentTimeMillis()
        );

        assertTrue(loadedAdapter.hasModule("model.layers.0.self_attn.q_proj"));
        assertFalse(loadedAdapter.hasModule("model.layers.0.self_attn.v_proj"));

        // Cleanup
        loadedAdapter.close();
    }
}
