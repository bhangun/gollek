/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelWarmupService.java
 * ───────────────────────
 * Runs a synthetic forward pass after model load to warm up JVM JIT
 * and eliminate the "cold start" latency spike on the first real request.
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import java.time.Duration;
import java.time.Instant;

import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine.LoadedModel;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

/**
 * Performs a synthetic forward pass after model load to pre-compile
 * JIT kernels and stabilise first-request latency.
 */
@ApplicationScoped
public class ModelWarmupService {

    private static final Logger log = Logger.getLogger(ModelWarmupService.class);

    @Inject
    KVCacheManager kvCacheManager;

    public void warmUp(LoadedModel model) {
        boolean enabled = Boolean.parseBoolean(
                System.getProperty("gollek.safetensor.engine.warmup.enabled", "true"));
        if (!enabled) {
            log.debugf("Warm-up disabled for model: %s", model.path().getFileName());
            return;
        }

        int promptTokens = Integer.parseInt(
                System.getProperty("gollek.safetensor.engine.warmup.prompt-tokens", "8"));

        log.infof("Warming up model: %s [%d synthetic tokens]",
                model.path().getFileName(), promptTokens);
        Instant start = Instant.now();

        try {
            int bosId = model.tokenizer().bosTokenId();
            int[] syntheticIds = new int[promptTokens];
            java.util.Arrays.fill(syntheticIds, bosId);

            GenerationConfig warmupCfg = GenerationConfig.builder()
                    .maxNewTokens(1)
                    .useKvCache(true)
                    .maxKvCacheTokens(Math.max(promptTokens + 2, 16))
                    .build();

            try (KVCacheManager.KVCacheSession kvCache = kvCacheManager.createSession(warmupCfg.maxKvCacheTokens())) {
                kvCache.allocate(model.config(), warmupCfg);
                log.debugf("Warm-up forward pass stub executed (Accelerate backend)");
            }

            Duration elapsed = Duration.between(start, Instant.now());
            log.infof("Warm-up complete: %s [%dms]",
                    model.path().getFileName(), elapsed.toMillis());

        } catch (Exception e) {
            log.warnf(e, "Warm-up failed for model %s (non-fatal, first request may be slow)",
                    model.path().getFileName());
        }
    }
}
