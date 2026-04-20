/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationEngine.java
 * ───────────────────────
 * Model quantization support (GPTQ, FP8).
 */
package tech.kayys.gollek.safetensor.quantization;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.quantizer.FP8Quantizer;
import tech.kayys.gollek.safetensor.quantization.quantizer.GPTQQuantizer;
import tech.kayys.gollek.safetensor.quantization.quantizer.INT8Quantizer;
import tech.kayys.gollek.safetensor.quantization.quantizer.Quantizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Quantization engine for model compression.
 * <p>
 * Supports multiple quantization strategies:
 * <ul>
 * <li>INT4 - 4-bit integer quantization using GPTQ algorithm</li>
 * <li>INT8 - 8-bit integer quantization with per-channel scaling</li>
 * <li>FP8 - 8-bit floating point quantization (E4M3/E5M2)</li>
 * </ul>
 *
 * @author Bhangun
 * @version 1.0.0
 */
@ApplicationScoped
public class QuantizationEngine {

    private static final Logger log = Logger.getLogger(QuantizationEngine.class);

    /**
     * Quantization strategy.
     */
    public enum QuantStrategy {
        /**
         * No quantization (passthrough).
         */
        NONE,
        /**
         * 4-bit integer quantization (GPTQ).
         * Best for CPU inference with significant memory reduction.
         */
        INT4,
        /**
         * 8-bit integer quantization.
         * Good balance between quality and performance.
         */
        INT8,
        /**
         * 8-bit floating point quantization.
         * Best for GPU inference with FP8 tensor cores.
         */
        FP8
    }

    /**
     * Quantization progress information.
     */
    public record QuantProgress(
            String tensorName,
            int currentTensor,
            int totalTensors,
            double percentComplete,
            String phase,
            String message) {
    }

    private final Map<String, Quantizer> quantizers = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Map<Path, QuantResult> resultsCache = new ConcurrentHashMap<>();

    /**
     * Initialize quantization engine with default quantizers.
     */
    public QuantizationEngine() {
        registerQuantizer(QuantStrategy.INT4, new GPTQQuantizer());
        registerQuantizer(QuantStrategy.INT8, new INT8Quantizer());
        registerQuantizer(QuantStrategy.FP8, new FP8Quantizer());
    }

    /**
     * Register a custom quantizer.
     *
     * @param strategy strategy this quantizer handles
     * @param quantizer quantizer implementation
     */
    public void registerQuantizer(QuantStrategy strategy, Quantizer quantizer) {
        quantizers.put(strategy.name().toLowerCase(), quantizer);
        log.infof("Registered quantizer for strategy: %s", strategy);
    }

    /**
     * Quantize a model asynchronously.
     *
     * @param modelPath     path to the model directory
     * @param outputPath    path for the quantized output
     * @param strategy      quantization strategy
     * @param config        quantization configuration
     * @return Uni emitting quantization result
     */
    public Uni<QuantResult> quantizeAsync(Path modelPath, Path outputPath, QuantStrategy strategy, QuantConfig config) {
        return Uni.createFrom().<QuantResult>emitter(em -> {
            try {
                QuantResult result = quantize(modelPath, outputPath, strategy, config);
                em.complete(result);
            } catch (Exception e) {
                em.fail(e);
            }
        }).runSubscriptionOn(executor);
    }

    /**
     * Quantize a model with progress streaming.
     *
     * @param modelPath     path to the model directory
     * @param outputPath    path for the quantized output
     * @param strategy      quantization strategy
     * @param config        quantization configuration
     * @return Multi emitting progress updates and final result
     */
    public Multi<Object> quantizeWithProgress(Path modelPath, Path outputPath, QuantStrategy strategy,
            QuantConfig config) {
        return Multi.createFrom().emitter(em -> {
            Instant startTime = Instant.now();

            try {
                // Validate input
                validateModelPath(modelPath);

                // Create output directory
                Files.createDirectories(outputPath);

                // Get quantizer
                Quantizer quantizer = getQuantizer(strategy);
                if (quantizer == null) {
                    throw new IllegalStateException("No quantizer found for strategy: " + strategy);
                }

                // Load model weights
                em.emit(new QuantProgress("", 0, 0, 0.0, "LOADING", "Loading model weights..."));
                Map<String, AccelTensor> weights = loadWeights(modelPath);
                long originalSize = calculateTotalSize(weights);

                em.emit(new QuantProgress("", 0, weights.size(), 0.0, "PREPARING",
                        "Preparing for quantization..."));

                // Perform quantization with progress
                Map<String, AccelTensor> quantizedWeights = new HashMap<>();
                int totalTensors = weights.size();
                int currentTensor = 0;

                for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
                    String tensorName = entry.getKey();
                    AccelTensor tensor = entry.getValue();

                    em.emit(new QuantProgress(
                            tensorName,
                            currentTensor,
                            totalTensors,
                            (double) currentTensor / totalTensors * 100.0,
                            "QUANTIZING",
                            String.format("Quantizing tensor %d/%d: %s", currentTensor + 1, totalTensors, tensorName)));

                    AccelTensor quantizedTensor = quantizer.quantizeTensor(tensor, config);
                    quantizedWeights.put(tensorName, quantizedTensor);

                    currentTensor++;
                }

                em.emit(new QuantProgress("", totalTensors, totalTensors, 100.0, "SAVING", "Saving quantized model..."));

                // Save quantized weights
                saveWeights(outputPath, quantizedWeights);

                // Calculate statistics
                long quantizedSize = calculateTotalSize(quantizedWeights);
                QuantStats stats = QuantStats.builder()
                        .originalSizeBytes(originalSize)
                        .quantizedSizeBytes(quantizedSize)
                        .tensorCount(totalTensors)
                        .paramCount(countParameters(weights))
                        .startTime(startTime)
                        .endTime(Instant.now())
                        .status(QuantStats.Status.COMPLETED)
                        .build();

                QuantResult result = QuantResult.success(outputPath, stats, config);
                resultsCache.put(outputPath, result);

                em.emit(result);
                em.complete();

            } catch (Exception e) {
                log.errorf(e, "Quantization failed");
                em.emit(QuantResult.failure(e.getMessage(), config));
                em.fail(e);
            }
        }).runSubscriptionOn(executor);
    }

    /**
     * Quantize a model.
     *
     * @param modelPath  path to the model directory
     * @param outputPath path for the quantized output
     * @param strategy   quantization strategy
     * @return quantization result
     */
    public QuantResult quantize(Path modelPath, Path outputPath, QuantStrategy strategy) {
        return quantize(modelPath, outputPath, strategy, createDefaultConfig(strategy));
    }

    /**
     * Quantize a model with custom configuration.
     *
     * @param modelPath  path to the model directory
     * @param outputPath path for the quantized output
     * @param strategy   quantization strategy
     * @param config     quantization configuration
     * @return quantization result
     */
    public QuantResult quantize(Path modelPath, Path outputPath, QuantStrategy strategy, QuantConfig config) {
        log.infof("Starting quantization: %s -> %s (strategy=%s)", modelPath, outputPath, strategy);

        Instant startTime = Instant.now();
        try {
            // Validate input
            validateModelPath(modelPath);

            // Create output directory
            Files.createDirectories(outputPath);

            // Get quantizer
            Quantizer quantizer = getQuantizer(strategy);
            if (quantizer == null) {
                throw new IllegalStateException("No quantizer found for strategy: " + strategy);
            }

            // Load model weights
            log.infof("Loading model weights from: %s", modelPath);
            Map<String, AccelTensor> weights = loadWeights(modelPath);
            long originalSize = calculateTotalSize(weights);
            log.infof("Loaded %d tensors (%s)", weights.size(), QuantStats.formatSize(originalSize));

            // Perform quantization
            log.infof("Quantizing with strategy: %s", strategy);
            Map<String, AccelTensor> quantizedWeights = new HashMap<>();
            int totalTensors = weights.size();
            int currentTensor = 0;

            for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
                String tensorName = entry.getKey();
                AccelTensor tensor = entry.getValue();

                if (currentTensor % 10 == 0) {
                    log.infof("Progress: %d/%d tensors (%.1f%%)", currentTensor, totalTensors,
                            (double) currentTensor / totalTensors * 100.0);
                }

                AccelTensor quantizedTensor = quantizer.quantizeTensor(tensor, config);
                quantizedWeights.put(tensorName, quantizedTensor);
                tensor.close(); // Free original tensor memory

                currentTensor++;
            }

            // Save quantized weights
            log.infof("Saving quantized model to: %s", outputPath);
            saveWeights(outputPath, quantizedWeights);

            // Calculate statistics
            long quantizedSize = calculateTotalSize(quantizedWeights);
            QuantStats stats = QuantStats.builder()
                    .originalSizeBytes(originalSize)
                    .quantizedSizeBytes(quantizedSize)
                    .tensorCount(totalTensors)
                    .paramCount(countParameters(weights))
                    .startTime(startTime)
                    .endTime(Instant.now())
                    .status(QuantStats.Status.COMPLETED)
                    .build();

            QuantResult result = QuantResult.success(outputPath, stats, config);
            resultsCache.put(outputPath, result);

            log.infof("Quantization complete: %s -> %s", QuantStats.formatSize(originalSize),
                    QuantStats.formatSize(quantizedSize));
            log.infof("Compression ratio: %.2fx", stats.getCompressionRatio());

            return result;

        } catch (Exception e) {
            log.errorf(e, "Quantization failed");
            QuantStats stats = QuantStats.builder()
                    .startTime(startTime)
                    .endTime(Instant.now())
                    .status(QuantStats.Status.FAILED)
                    .errorMessage(e.getMessage())
                    .build();
            return QuantResult.failure(e.getMessage(), config);
        }
    }

    /**
     * Load a quantized model for inference.
     *
     * @param quantizedModelPath path to quantized model
     * @param strategy           quantization strategy
     * @return dequantized weights map
     */
    public Map<String, AccelTensor> loadQuantizedModel(Path quantizedModelPath, QuantStrategy strategy) {
        log.infof("Loading quantized model: %s (strategy=%s)", quantizedModelPath, strategy);

        try {
            Map<String, AccelTensor> quantizedWeights = loadWeights(quantizedModelPath);
            Quantizer quantizer = getQuantizer(strategy);

            if (quantizer == null) {
                throw new IllegalStateException("No quantizer found for strategy: " + strategy);
            }

            // Dequantize weights
            Map<String, AccelTensor> dequantizedWeights = new HashMap<>();
            for (Map.Entry<String, AccelTensor> entry : quantizedWeights.entrySet()) {
                AccelTensor dequantized = quantizer.dequantizeTensor(entry.getValue(), createDefaultConfig(strategy));
                dequantizedWeights.put(entry.getKey(), dequantized);
            }

            log.infof("Loaded %d quantized tensors", dequantizedWeights.size());
            return dequantizedWeights;

        } catch (IOException e) {
            throw new RuntimeException("Failed to load quantized model", e);
        }
    }

    /**
     * Get cached quantization result.
     *
     * @param modelPath path to quantized model
     * @return cached result or null
     */
    public QuantResult getCachedResult(Path modelPath) {
        return resultsCache.get(modelPath);
    }

    /**
     * Clear the results cache.
     */
    public void clearCache() {
        resultsCache.clear();
        log.info("Quantization results cache cleared");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    private Quantizer getQuantizer(QuantStrategy strategy) {
        return quantizers.get(strategy.name().toLowerCase());
    }

    private void validateModelPath(Path modelPath) {
        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("Model path does not exist: " + modelPath);
        }
        if (!Files.isDirectory(modelPath)) {
            throw new IllegalArgumentException("Model path is not a directory: " + modelPath);
        }
    }

    private Map<String, AccelTensor> loadWeights(Path modelPath) throws IOException {
        // TODO: Implement SafeTensor reader
        // For now, return empty map - actual implementation would use SafeTensor reader
        log.warnf("SafeTensor reader not yet implemented - returning empty weights");
        return new HashMap<>();
    }

    private void saveWeights(Path outputPath, Map<String, AccelTensor> weights) throws IOException {
        // TODO: Implement SafeTensor writer
        // For now, no-op - actual implementation would use SafeTensor writer
        log.warnf("SafeTensor writer not yet implemented - weights not saved");
    }

    private long calculateTotalSize(Map<String, AccelTensor> weights) {
        return weights.values().stream()
                .mapToLong(this::estimateTensorSize)
                .sum();
    }

    private long estimateTensorSize(AccelTensor tensor) {
        // Estimate tensor size in bytes
        // Actual implementation would get dtype and shape from tensor
        return tensor != null ? 1024L : 0L; // Placeholder
    }

    private long countParameters(Map<String, AccelTensor> weights) {
        return weights.values().stream()
                .mapToLong(this::estimateParamCount)
                .sum();
    }

    private long estimateParamCount(AccelTensor tensor) {
        // Estimate parameter count
        // Actual implementation would get shape from tensor
        return tensor != null ? 256L : 0L; // Placeholder
    }

    private QuantConfig createDefaultConfig(QuantStrategy strategy) {
        return switch (strategy) {
            case INT4 -> QuantConfig.int4Gptq();
            case INT8 -> QuantConfig.int8();
            case FP8 -> QuantConfig.fp8();
            default -> new QuantConfig();
        };
    }

    /**
     * Shutdown the engine and release resources.
     */
    public void shutdown() {
        executor.shutdown();
        clearCache();
        log.info("Quantization engine shutdown complete");
    }
}
