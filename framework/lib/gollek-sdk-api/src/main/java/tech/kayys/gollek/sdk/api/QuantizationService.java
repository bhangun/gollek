/*
 * Gollek Inference Engine — SDK Integration
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationService.java
 * ────────────────────────
 * High-level quantization service integrating all quantizer modules
 * with the safetensor infrastructure and SDK.
 */
package tech.kayys.gollek.sdk.api;

import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantResult;
import tech.kayys.gollek.safetensor.quantization.QuantStats;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.safetensor.quantization.QuantizerRegistry;
import tech.kayys.gollek.safetensor.quantization.quantizer.Quantizer;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.jboss.logging.Logger;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * High-level quantization service for the Gollek SDK.
 * <p>
 * Provides a unified interface to all quantization backends:
 * <ul>
 * <li><b>GPTQ</b> - Hessian-based per-layer quantization (best quality)</li>
 * <li><b>AWQ</b> - Activation-aware weight quantization (fast)</li>
 * <li><b>AutoRound</b> - Optimization-based rounding (balanced)</li>
 * <li><b>TurboQuant</b> - Online vector quantization (edge-optimized)</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * QuantizationService qs = QuantizationService.getInstance();
 *
 * // Quantize with GPTQ
 * QuantResult result = qs.quantizeGptq(
 *     Path.of("/models/llama-7b-fp16"),
 *     Path.of("/models/llama-7b-int4")
 * );
 *
 * // Quantize with progress streaming
 * qs.quantizeWithProgress(modelPath, outputPath, QuantConfig.int4Gptq())
 *   .subscribe().with(
 *       item -> System.out.println(item),
 *       error -> System.err.println(error),
 *       () -> System.out.println("Done!")
 *   );
 * }</pre>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class QuantizationService {

    private static final Logger log = Logger.getLogger(QuantizationService.class);

    @Inject
    QuantizationEngine engine;

    /**
     * Get the singleton instance (for non-CDI usage).
     *
     * @return quantization service instance
     */
    public static QuantizationService getInstance() {
        return LazyHolder.INSTANCE;
    }

    private static class LazyHolder {
        static final QuantizationService INSTANCE = new QuantizationService();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Quantizer Discovery
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get all available quantizer names.
     *
     * @return set of quantizer names
     */
    public Set<String> getAvailableQuantizers() {
        return QuantizerRegistry.getNames();
    }

    /**
     * Get a specific quantizer by name.
     *
     * @param name quantizer name
     * @return quantizer instance, or null if not found
     */
    public Quantizer getQuantizer(String name) {
        return QuantizerRegistry.get(name);
    }

    /**
     * Get quantizer statistics.
     *
     * @return map of quantizer metadata
     */
    public Map<String, String> getQuantizerStats() {
        return QuantizerRegistry.getStats();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // High-Level Quantization API
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Quantize a model using GPTQ (4-bit).
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @return quantization result
     */
    public QuantResult quantizeGptq(Path modelPath, Path outputPath) {
        return quantizeGptq(modelPath, outputPath, QuantConfig.int4Gptq());
    }

    /**
     * Quantize a model using GPTQ with custom config.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @param config     GPTQ configuration
     * @return quantization result
     */
    public QuantResult quantizeGptq(Path modelPath, Path outputPath, QuantConfig config) {
        log.infof("GPTQ quantization: %s -> %s", modelPath, outputPath);
        return engine.quantize(modelPath, outputPath, QuantizationEngine.QuantStrategy.INT4, config);
    }

    /**
     * Quantize a model using INT8.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @return quantization result
     */
    public QuantResult quantizeInt8(Path modelPath, Path outputPath) {
        return engine.quantize(modelPath, outputPath, QuantizationEngine.QuantStrategy.INT8);
    }

    /**
     * Quantize a model using FP8.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @return quantization result
     */
    public QuantResult quantizeFp8(Path modelPath, Path outputPath) {
        return engine.quantize(modelPath, outputPath, QuantizationEngine.QuantStrategy.FP8);
    }

    /**
     * Quantize a model with automatic strategy selection.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @param config     quantization configuration
     * @return quantization result
     */
    public QuantResult quantize(Path modelPath, Path outputPath, QuantConfig config) {
        Quantizer quantizer = QuantizerRegistry.selectBest(config);
        if (quantizer == null) {
            throw new IllegalStateException("No suitable quantizer found for config: " + config);
        }
        log.infof("Selected quantizer: %s", quantizer.getName());
        return engine.quantize(modelPath, outputPath, config.getStrategy(), config);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Reactive API with Progress Streaming
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Quantize a model asynchronously.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @param config     quantization configuration
     * @return Uni emitting quantization result
     */
    public Uni<QuantResult> quantizeAsync(Path modelPath, Path outputPath, QuantConfig config) {
        return engine.quantizeAsync(modelPath, outputPath, QuantizationEngine.QuantStrategy.INT4, config);
    }

    /**
     * Quantize a model with progress streaming.
     *
     * @param modelPath  path to FP16/FP32 model
     * @param outputPath path for quantized output
     * @param config     quantization configuration
     * @return Multi emitting progress updates and final result
     */
    public Multi<Object> quantizeWithProgress(Path modelPath, Path outputPath, QuantConfig config) {
        return engine.quantizeWithProgress(modelPath, outputPath, QuantizationEngine.QuantStrategy.INT4, config);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Model Loading
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Load a quantized model for inference.
     *
     * @param quantizedModelPath path to quantized model
     * @param strategy           quantization strategy
     * @return dequantized weights
     */
    public java.util.Map<String, tech.kayys.gollek.inference.libtorch.core.TorchTensor> loadQuantizedModel(
            Path quantizedModelPath, QuantizationEngine.QuantStrategy strategy) {
        return engine.loadQuantizedModel(quantizedModelPath, strategy);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Cache Management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get cached quantization result.
     *
     * @param modelPath path to quantized model
     * @return cached result or null
     */
    public QuantResult getCachedResult(Path modelPath) {
        return engine.getCachedResult(modelPath);
    }

    /**
     * Clear the quantization results cache.
     */
    public void clearCache() {
        engine.clearCache();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Shutdown the quantization service.
     */
    public void shutdown() {
        engine.shutdown();
    }
}
