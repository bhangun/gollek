/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ──────────────────────────
 * Loads SafeTensor model weights using FFM mmap and bridges them to
 * AccelTensor — pure Java + Apple Accelerate. No LibTorch dependency.
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.alkhawarizm.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.warmup.ModelWarmupService;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.Map;
import java.util.Collection;

@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {
    @Inject
    Instance<Object> metalBackend;

    /**
     * Direct SafeTensor inference engine using AccelTensor + Apple Accelerate.
     * No LibTorch dependency.
     */

    private static final Logger log = Logger.getLogger(DirectInferenceEngine.class);

    @Inject
    SafetensorLoaderFacade safetensorLoader;

    @Inject
    AccelWeightBridge bridge;

    @Inject
    QuantizationEngine quantizationEngine;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    DirectForwardPass forwardPass;
    @Inject
    TokenSampler tokenSampler;
    private DirectInferenceComponentGraph componentGraph;

    @Inject
    ModelFamilyPluginRegistry archRegistry;

    @Inject
    KVCacheManager kvCacheManager;

    @Inject
    ModelWarmupService modelWarmupService;

    private ModelFamilyPluginRegistry architectureRegistry() {
        archRegistry = DirectInferenceDependencies.resolve(
                archRegistry, ModelFamilyPluginRegistry.class, "ModelFamilyPluginRegistry");
        return archRegistry;
    }

    private DirectForwardPass forwardPass() {
        forwardPass = DirectInferenceDependencies.resolve(
                forwardPass, DirectForwardPass.class, "DirectForwardPass");
        return forwardPass;
    }

    private TokenSampler tokenSampler() {
        tokenSampler = DirectInferenceDependencies.resolve(tokenSampler, TokenSampler.class, "TokenSampler");
        return tokenSampler;
    }

    private SafetensorLoaderFacade safetensorLoader() {
        safetensorLoader = DirectInferenceDependencies.resolve(
                safetensorLoader, SafetensorLoaderFacade.class, "SafetensorLoaderFacade");
        return safetensorLoader;
    }

    private AccelWeightBridge weightBridge() {
        bridge = DirectInferenceDependencies.resolve(bridge, AccelWeightBridge.class, "AccelWeightBridge");
        return bridge;
    }

    private ObjectMapper resolvedObjectMapper() {
        objectMapper = DirectInferenceDependencies.resolve(objectMapper, ObjectMapper.class, "ObjectMapper");
        return objectMapper;
    }

    private DirectInferenceComponentGraph components() {
        if (componentGraph == null) {
            componentGraph = new DirectInferenceComponentGraph(
                    () -> metalBackend,
                    log,
                    this::safetensorLoader,
                    this::weightBridge,
                    this::resolvedObjectMapper,
                    this::getQuantizationEngine,
                    this::architectureRegistry,
                    this::forwardPass,
                    this::tokenSampler,
                    this::getKVCacheManager,
                    () -> modelWarmupService);
        }
        return componentGraph;
    }

    public record DirectGenerationTrace(
            InferenceResponse response,
            long[] inputIds,
            long[] generatedTokenIds) {
    }

    public record DirectConversationTrace(
            InferenceResponse response,
            long[] inputIds,
            long[] generatedTokenIds,
            KVCacheManager.KVCacheSession kvCacheSession) {
    }

    /** Compatibility type; loaded-model state lives in {@link DirectLoadedModel}. */
    public static class LoadedModel extends DirectLoadedModel {
        public LoadedModel(Path path, Map<String, AccelTensor> weights,
                Tokenizer tokenizer, String key,
                boolean quantized, QuantizationEngine.QuantStrategy quantStrategy,
                String quantCacheState, Path quantCachePath,
                ModelConfig config, ModelArchitecture architecture, ModelRuntimeTraits runtimeTraits,
                Arena weightArena) {
            super(path, weights, tokenizer, key, quantized, quantStrategy, quantCacheState, quantCachePath,
                    config, architecture, runtimeTraits, weightArena);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String loadModel(Path modelPath) {
        return loadModel(modelPath, null, QuantizationEngine.QuantStrategy.NONE);
    }

    public String loadModel(Path modelPath, Path adapterPath, String quantStrategy) {
        tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine.QuantStrategy resolvedQuantStrategy = 
                tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine.QuantStrategy.NONE;
        if (quantStrategy != null && !quantStrategy.isBlank()) {
            try {
                resolvedQuantStrategy = tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine.QuantStrategy.valueOf(quantStrategy.toUpperCase());
            } catch (Exception e) {
                // Ignore, fallback to NONE
            }
        }
        return loadModel(modelPath, adapterPath, resolvedQuantStrategy);
    }

    public String loadModel(Path modelPath, Path adapterPath,
            QuantizationEngine.QuantStrategy quantStrategy) {
        return components().modelStore().load(modelPath, quantStrategy);
    }

    public Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return components().syncApi().generate(prompt, modelPath, cfg);
    }

    public long[] encodePrompt(String prompt, Path modelPath) {
        return components().syncApi().encodePrompt(prompt, modelPath);
    }

    public Uni<DirectGenerationTrace> generateWithTrace(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return components().syncApi().generateWithTrace(inputIds, modelPath, cfg);
    }

    public Uni<InferenceResponse> generate(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return components().syncApi().generate(inputIds, modelPath, cfg);
    }

    public Uni<DirectConversationTrace> generateWithConversationTrace(long[] inputIds, Path modelPath,
            GenerationConfig cfg) {
        return components().syncApi().generateWithConversationTrace(inputIds, modelPath, cfg);
    }

    public Uni<DirectConversationTrace> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg) {
        return generateContinuationWithConversationTrace(
                fullInputIds,
                cachedPrefixTokens,
                session,
                modelPath,
                cfg,
                null);
    }

    public Uni<DirectConversationTrace> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Integer replayTokenId) {
        return components().syncApi().generateContinuationWithConversationTrace(
                fullInputIds, cachedPrefixTokens, session, modelPath, cfg, replayTokenId);
    }

    public Multi<InferenceResponse> generateStreamWithConversationTrace(
            long[] inputIds,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete) {
        return components().streamingApi().generateWithConversationTrace(inputIds, modelPath, cfg, onComplete);
    }

    public Multi<InferenceResponse> generateContinuationStreamWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete) {
        return generateContinuationStreamWithConversationTrace(
                fullInputIds,
                cachedPrefixTokens,
                session,
                modelPath,
                cfg,
                onComplete,
                null);
    }

    public Multi<InferenceResponse> generateContinuationStreamWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete,
            Integer replayTokenId) {
        return components().streamingApi().generateContinuationWithConversationTrace(
                fullInputIds, cachedPrefixTokens, session, modelPath, cfg, onComplete, replayTokenId);
    }

    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return components().streamingApi().generate(prompt, modelPath, cfg);
    }

    public Multi<InferenceResponse> generateStream(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return components().streamingApi().generate(inputIds, modelPath, cfg);
    }

    @Override
    public void unloadModel(Path modelPath) {
        components().modelStore().unload(modelPath);
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(Path modelPath) {
        return components().modelStore().find(modelPath);
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(String key) {
        return components().modelStore().findByKey(key);
    }

    public boolean isLoaded(Path modelPath) {
        return components().modelStore().contains(modelPath);
    }

    public Collection<LoadedModel> listLoadedModels() {
        return components().modelStore().snapshot();
    }

    public QuantizationEngine getQuantizationEngine() {
        quantizationEngine = DirectInferenceDependencies.resolve(
                quantizationEngine, QuantizationEngine.class, "QuantizationEngine");
        return quantizationEngine;
    }

    public KVCacheManager getKVCacheManager() {
        kvCacheManager = DirectInferenceDependencies.resolve(kvCacheManager, KVCacheManager.class, "KVCacheManager");
        return kvCacheManager;
    }

}
