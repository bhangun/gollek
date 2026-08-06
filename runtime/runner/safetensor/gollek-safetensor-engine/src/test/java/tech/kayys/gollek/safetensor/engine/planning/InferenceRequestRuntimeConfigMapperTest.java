/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendCapabilities;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InferenceRequestRuntimeConfigMapperTest {
    private final InferenceRequestRuntimeConfigMapper mapper = new InferenceRequestRuntimeConfigMapper();

    @Test
    void generationConfigPreservesProviderSamplingKnobs() {
        ProviderRequest request = request()
                .maxTokens(64)
                .temperature(0.8)
                .topK(16)
                .topP(0.9)
                .repeatPenalty(1.25)
                .parameter("min_p", 0.05)
                .parameter("seed", 42L)
                .parameter("kv_cache_quant", "int8")
                .build();

        GenerationConfig config = mapper.generationConfig(request, capabilities(true));

        assertEquals(64, config.maxNewTokens());
        assertEquals(GenerationConfig.SamplingStrategy.TOP_K_TOP_P, config.strategy());
        assertEquals(0.8f, config.temperature(), 1.0e-6f);
        assertEquals(16, config.topK());
        assertEquals(0.9f, config.topP(), 1.0e-6f);
        assertEquals(0.05f, config.minP(), 1.0e-6f);
        assertEquals(1.25f, config.repetitionPenalty(), 1.0e-6f);
        assertEquals(GenerationConfig.KvCacheQuantization.INT8, config.kvCacheQuant());
        assertEquals(42L, config.seed());
    }

    @Test
    void zeroMaxTokensFallsBackToPlannerDefault() {
        ProviderRequest request = request()
                .maxTokens(0)
                .build();

        GenerationConfig config = mapper.generationConfig(request, capabilities(true));

        assertEquals(256, config.maxNewTokens());
    }

    @Test
    void unsupportedKvQuantizationFallsBackToNone() {
        ProviderRequest request = request()
                .parameter("kv_cache_quant", "int4")
                .build();

        GenerationConfig config = mapper.generationConfig(request, capabilities(false));

        assertEquals(GenerationConfig.KvCacheQuantization.NONE, config.kvCacheQuant());
    }

    @Test
    void turboKvQuantizationMapsToPackedInt4() {
        assertEquals(
                GenerationConfig.KvCacheQuantization.INT4,
                InferenceRequestRuntimeConfigMapper.normalizeKvQuantization("turbo", capabilities(true)));
    }

    @Test
    void samplingStrategyResolvesDeterministicAndFilteredModes() {
        assertEquals(
                GenerationConfig.SamplingStrategy.GREEDY,
                InferenceRequestRuntimeConfigMapper.resolveSamplingStrategy(0.0f, 40, 0.9f));
        assertEquals(
                GenerationConfig.SamplingStrategy.GREEDY,
                InferenceRequestRuntimeConfigMapper.resolveSamplingStrategy(0.7f, 1, 0.9f));
        assertEquals(
                GenerationConfig.SamplingStrategy.TOP_P,
                InferenceRequestRuntimeConfigMapper.resolveSamplingStrategy(0.7f, 0, 0.9f));
        assertEquals(
                GenerationConfig.SamplingStrategy.TOP_K,
                InferenceRequestRuntimeConfigMapper.resolveSamplingStrategy(0.7f, 40, 1.0f));
    }

    @Test
    void audioConfigParsesFormatCaseInsensitively() {
        ProviderRequest request = request()
                .temperature(0.35)
                .parameter("output_format", "mP3")
                .build();

        AudioConfig config = mapper.audioConfig(request);

        assertEquals(AudioConfig.Format.MP3, config.getFormat());
        assertEquals(0.35f, config.getTemperature(), 1.0e-6f);
    }

    @Test
    void audioConfigFallsBackToWavForInvalidFormat() {
        ProviderRequest request = request()
                .parameter("output_format", "not-a-real-format")
                .build();

        AudioConfig config = mapper.audioConfig(request);

        assertEquals(AudioConfig.Format.WAV, config.getFormat());
    }

    private static ProviderRequest.Builder request() {
        return ProviderRequest.builder()
                .model("unit-test-model")
                .message(Message.user("hello"));
    }

    private static TextExecutionBackendCapabilities capabilities(boolean supportsKvCacheQuantization) {
        return new TextExecutionBackendCapabilities(
                true,
                true,
                supportsKvCacheQuantization,
                false,
                false,
                false,
                true);
    }
}
