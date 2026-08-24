/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import tech.kayys.alkhawarizm.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Builds a direct loaded model from model files and engine services.
 */
final class DirectModelLoader {
    private static final Logger LOG = Logger.getLogger(DirectModelLoader.class);

    private final Supplier<SafetensorLoaderFacade> safetensorLoader;
    private final Supplier<AccelWeightBridge> weightBridge;
    private final Supplier<ObjectMapper> objectMapper;
    private final Supplier<QuantizationEngine> quantizationEngine;
    private final Supplier<ModelFamilyPluginRegistry> architectureRegistry;

    DirectModelLoader(Supplier<SafetensorLoaderFacade> safetensorLoader,
            Supplier<AccelWeightBridge> weightBridge,
            Supplier<ObjectMapper> objectMapper,
            Supplier<QuantizationEngine> quantizationEngine,
            Supplier<ModelFamilyPluginRegistry> architectureRegistry) {
        this.safetensorLoader = Objects.requireNonNull(safetensorLoader, "safetensorLoader");
        this.weightBridge = Objects.requireNonNull(weightBridge, "weightBridge");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.quantizationEngine = Objects.requireNonNull(quantizationEngine, "quantizationEngine");
        this.architectureRegistry = Objects.requireNonNull(architectureRegistry, "architectureRegistry");
    }

    DirectInferenceEngine.LoadedModel load(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy) {
        QuantizationEngine.QuantStrategy effectiveQuantStrategy = normalizeQuantStrategy(quantStrategy);

        Arena weightArena = Arena.ofAuto();
        ModelConfig config = loadConfig(modelPath);
        ModelArchitecture architecture = null;
        ModelFamilyPlugin plugin = architectureRegistry.get().all().stream().filter(p -> p.architectureAdapters().stream().anyMatch(a -> a.supportedArchClassNames().contains(config.primaryArchitecture()))).findFirst().orElse(null);
        if (plugin != null) {
            architecture = plugin.architectureAdapters().stream().filter(a -> a.supportedArchClassNames().contains(config.primaryArchitecture())).findFirst().orElse(null);
        }
        ModelRuntimeTraits runtimeTraits = plugin != null ? plugin.runtimeTraits(config) : ModelRuntimeTraits.fallbackFromConfig(config);
        WeightLoadResult weightLoadResult = loadWeights(modelPath, effectiveQuantStrategy, config, runtimeTraits);
        Map<String, AccelTensor> weights = weightLoadResult.weights();
        Tokenizer tokenizer = loadTokenizerWithContext(modelPath);
        String key = modelKey(modelPath, effectiveQuantStrategy);

        return new DirectInferenceEngine.LoadedModel(modelPath, weights, tokenizer, key,
                effectiveQuantStrategy != QuantizationEngine.QuantStrategy.NONE, effectiveQuantStrategy,
                weightLoadResult.quantCacheState(), weightLoadResult.quantCachePath(),
                config, architecture, runtimeTraits, weightArena);
    }

    static QuantizationEngine.QuantStrategy normalizeQuantStrategy(QuantizationEngine.QuantStrategy quantStrategy) {
        return quantStrategy == null ? QuantizationEngine.QuantStrategy.NONE : quantStrategy;
    }

    static String modelKey(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy) {
        QuantizationEngine.QuantStrategy effectiveQuantStrategy = normalizeQuantStrategy(quantStrategy);
        String baseName = modelPath.getFileName().toString();
        if (effectiveQuantStrategy == QuantizationEngine.QuantStrategy.NONE) {
            return baseName;
        }
        return baseName + "#" + effectiveQuantStrategy.name().toLowerCase();
    }

    private WeightLoadResult loadWeights(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy,
            ModelConfig config, ModelRuntimeTraits runtimeTraits) {
        LOG.debugf("DirectInferenceEngine: opening weights from [%s]", modelPath);
        System.err.printf("[Gollek] Phase 1/4: Opening safetensor shards from %s...%n", modelPath.getFileName());
        long t0 = System.currentTimeMillis();

        Map<String, AccelTensor> weights;
        try (SafetensorShardSession session = safetensorLoader.get().open(modelPath)) {
            System.err.printf("[Gollek] Phase 2/4: Bridging tensors (%.1fs)...%n", (System.currentTimeMillis() - t0) / 1000.0);
            weights = weightBridge.get().bridgeAll(session);
            System.err.printf("[Gollek] Phase 2/4: Bridged %d tensors (%.1fs)%n", weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
        } catch (Exception e) {
            LOG.errorf("Failed to load weights from %s: %s", modelPath, e.getMessage());
            throw new IllegalStateException("Failed to load weights: " + e.getMessage(), e);
        }

        String quantCacheState = "off";
        Path quantCachePath = null;
        if (quantStrategy == QuantizationEngine.QuantStrategy.NONE) {
            System.err.printf("[Gollek] Phase 3/4: Applying Metal F16 weight cache (%.1fs)...%n", (System.currentTimeMillis() - t0) / 1000.0);
            weights = new MetalF16WeightCache(safetensorLoader.get(), weightBridge.get())
                    .maybeUse(modelPath, weights, config, runtimeTraits);
        } else {
            System.err.printf("[Gollek] Phase 3/4: Quantizing weights with strategy=%s (%.1fs)...%n", quantStrategy, (System.currentTimeMillis() - t0) / 1000.0);
            QuantizationEngine.InferenceQuantizationResult quantizationResult = quantizationEngine.get()
                    .quantizeWeightsForInferenceDetailed(modelPath, weights, quantStrategy);
            weights = quantizationResult.weights();
            quantCacheState = quantizationResult.cacheState();
            quantCachePath = quantizationResult.cachePath();
        }

        System.err.printf("[Gollek] Phase 4/4: Expanding weight aliases (%.1fs)...%n", (System.currentTimeMillis() - t0) / 1000.0);
        WeightAliasExpander.applyCommonAliases(weights);
        System.err.printf("[Gollek] Model ready: %d weights total (%.1fs)%n", weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
        LOG.infof("DirectInferenceEngine: bridged %d tensors (Accelerate/FFM backend, strategy=%s)",
                weights.size(), quantStrategy);
        return new WeightLoadResult(weights, quantCacheState, quantCachePath);
    }

    private Tokenizer loadTokenizerWithContext(Path modelPath) {
        try {
            return loadTokenizer(modelPath);
        } catch (Exception e) {
            LOG.warnf("Failed to load tokenizer for [%s]", modelPath);
            throw new RuntimeException("Tokenizer loading failed", e);
        }
    }

    private Tokenizer loadTokenizer(Path modelPath) {
        try {
            return TokenizerFactory.load(modelPath, null);
        } catch (IOException e) {
            LOG.errorf(e, "Tokenizer loading failed for [%s]", modelPath);
            throw new RuntimeException("Tokenizer loading failed: " + e.getMessage(), e);
        }
    }

    private ModelConfig loadConfig(Path modelPath) {
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir != null) {
                return ModelConfig.fromDirectory(configDir, objectMapper.get());
            }
            return new ModelConfig();
        } catch (IOException e) {
            LOG.warnf(e, "DirectInferenceEngine: failed to read config.json near [%s]", modelPath);
            return new ModelConfig();
        }
    }

    private record WeightLoadResult(Map<String, AccelTensor> weights, String quantCacheState, Path quantCachePath) {
    }
}
