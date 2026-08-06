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
import java.util.Arrays;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine.LoadedModel;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;

/**
 * Performs a synthetic forward pass after model load to pre-compile
 * JIT kernels and stabilise first-request latency.
 */
@ApplicationScoped
public class ModelWarmupService {

    private static final Logger log = Logger.getLogger(ModelWarmupService.class);
    private static final String ENABLED_PROPERTY = "gollek.safetensor.engine.warmup.enabled";
    private static final String PROMPT_TOKENS_PROPERTY = "gollek.safetensor.engine.warmup.prompt-tokens";
    private static final String DECODE_TOKENS_PROPERTY = "gollek.safetensor.engine.warmup.decode-tokens";

    @Inject
    KVCacheManager kvCacheManager;

    @Inject
    DirectForwardPass forwardPass;

    public void warmUp(LoadedModel model) {
        boolean enabled = Boolean.parseBoolean(System.getProperty(ENABLED_PROPERTY, "false"));
        if (!enabled) {
            log.debugf("Warm-up disabled for model: %s", model.path().getFileName());
            return;
        }

        int promptTokens = Math.max(1, Integer.getInteger(PROMPT_TOKENS_PROPERTY, 8));
        int decodeTokens = Math.max(0, Integer.getInteger(DECODE_TOKENS_PROPERTY, 1));

        log.infof("Warming up model: %s [prefill=%d synthetic tokens, decode=%d token(s)]",
                model.path().getFileName(), promptTokens, decodeTokens);
        Instant start = Instant.now();

        try {
            long warmTokenId = warmTokenId(model);
            long[] syntheticIds = new long[promptTokens];
            Arrays.fill(syntheticIds, warmTokenId);

            GenerationConfig warmupCfg = GenerationConfig.builder()
                    .maxNewTokens(Math.max(1, decodeTokens))
                    .useKvCache(true)
                    .maxKvCacheTokens(Math.max(promptTokens + decodeTokens + 2, 16))
                    .build();
            ModelArchitecture arch = model.architecture();

            try (KVCacheManager.KVCacheSession kvCache = kvCacheManager.createSession(warmupCfg.maxKvCacheTokens())) {
                kvCache.allocate(model.config(), warmupCfg);
                AccelTensor prefillLogits = forwardPass.prefillLogitsTensor(
                        syntheticIds, model.weights(), model.config(), arch, kvCache);
                prefillLogits.close();

                int startPos = promptTokens;
                for (int i = 0; i < decodeTokens; i++) {
                    AccelTensor decodeLogits = forwardPass.decodeLogitsTensor(
                            warmTokenId, startPos + i, model.weights(), model.config(), arch, kvCache, true);
                    decodeLogits.close();
                }
            }

            Duration elapsed = Duration.between(start, Instant.now());
            log.infof("Warm-up complete: %s [%dms]",
                    model.path().getFileName(), elapsed.toMillis());

        } catch (Exception e) {
            log.warnf(e, "Warm-up failed for model %s (non-fatal, first request may be slow)",
                    model.path().getFileName());
        }
    }

    private static long warmTokenId(LoadedModel model) {
        int bos = model.tokenizer().bosTokenId();
        if (bos >= 0) {
            return bos;
        }
        int eos = model.tokenizer().eosTokenId();
        if (eos >= 0) {
            return eos;
        }
        int pad = model.tokenizer().padTokenId();
        if (pad >= 0) {
            return pad;
        }
        return 0L;
    }
}
