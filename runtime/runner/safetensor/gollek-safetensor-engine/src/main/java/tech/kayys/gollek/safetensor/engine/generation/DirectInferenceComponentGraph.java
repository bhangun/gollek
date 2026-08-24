/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.inject.Instance;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.warmup.ModelWarmupService;
import tech.kayys.alkhawarizm.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.quantization.bridge.AccelWeightBridge;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Owns lazy construction of direct inference collaborators.
 */
final class DirectInferenceComponentGraph {
    private final Supplier<Instance<Object>> metalBackend;
    private final Logger log;
    private final Supplier<SafetensorLoaderFacade> safetensorLoader;
    private final Supplier<AccelWeightBridge> weightBridge;
    private final Supplier<ObjectMapper> objectMapper;
    private final Supplier<QuantizationEngine> quantizationEngine;
    private final Supplier<ModelFamilyPluginRegistry> architectureRegistry;
    private final Supplier<DirectForwardPass> forwardPass;
    private final Supplier<TokenSampler> tokenSampler;
    private final Supplier<KVCacheManager> kvCacheManager;
    private final Supplier<ModelWarmupService> modelWarmupService;

    private DirectModelLoader modelLoader;
    private DirectInferenceModelStore modelStore;
    private DirectInferenceGenerationServices generationServices;
    private DirectInferenceSyncApi syncApi;
    private DirectInferenceStreamingApi streamingApi;

    DirectInferenceComponentGraph(Supplier<Instance<Object>> metalBackend,
            Logger log,
            Supplier<SafetensorLoaderFacade> safetensorLoader,
            Supplier<AccelWeightBridge> weightBridge,
            Supplier<ObjectMapper> objectMapper,
            Supplier<QuantizationEngine> quantizationEngine,
            Supplier<ModelFamilyPluginRegistry> architectureRegistry,
            Supplier<DirectForwardPass> forwardPass,
            Supplier<TokenSampler> tokenSampler,
            Supplier<KVCacheManager> kvCacheManager,
            Supplier<ModelWarmupService> modelWarmupService) {
        this.metalBackend = Objects.requireNonNull(metalBackend, "metalBackend");
        this.log = Objects.requireNonNull(log, "log");
        this.safetensorLoader = Objects.requireNonNull(safetensorLoader, "safetensorLoader");
        this.weightBridge = Objects.requireNonNull(weightBridge, "weightBridge");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.quantizationEngine = Objects.requireNonNull(quantizationEngine, "quantizationEngine");
        this.architectureRegistry = Objects.requireNonNull(architectureRegistry, "architectureRegistry");
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.tokenSampler = Objects.requireNonNull(tokenSampler, "tokenSampler");
        this.kvCacheManager = Objects.requireNonNull(kvCacheManager, "kvCacheManager");
        this.modelWarmupService = modelWarmupService;
    }

    DirectInferenceSyncApi syncApi() {
        if (syncApi == null) {
            syncApi = new DirectInferenceSyncApi(
                    metalBackend,
                    log,
                    (path, verbose, debugPrefix) -> modelStore().require(path, verbose, debugPrefix),
                    generationServices()::generationFlows);
        }
        return syncApi;
    }

    DirectInferenceStreamingApi streamingApi() {
        if (streamingApi == null) {
            streamingApi = new DirectInferenceStreamingApi(
                    metalBackend,
                    log,
                    generationServices()::streamingGenerationFlows);
        }
        return streamingApi;
    }

    DirectInferenceModelStore modelStore() {
        if (modelStore == null) {
            modelStore = new DirectInferenceModelStore(
                    this::modelLoader,
                    forwardPass,
                    modelWarmupService,
                    log);
        }
        return modelStore;
    }

    private DirectInferenceGenerationServices generationServices() {
        if (generationServices == null) {
            generationServices = new DirectInferenceGenerationServices(
                    forwardPass,
                    tokenSampler,
                    kvCacheManager,
                    (path, verbose, debugPrefix) -> modelStore().require(path, verbose, debugPrefix));
        }
        return generationServices;
    }

    private DirectModelLoader modelLoader() {
        if (modelLoader == null) {
            modelLoader = new DirectModelLoader(
                    safetensorLoader,
                    weightBridge,
                    objectMapper,
                    quantizationEngine,
                    architectureRegistry);
        }
        return modelLoader;
    }
}
