package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRTGemmaNativeRunnerHeuristicsTest {

    @AfterEach
    void clearRawPrefillThresholdOverride() {
        System.clearProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_MIN_TOKENS_PROPERTY);
        System.clearProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY);
        System.clearProperty(LiteRTGemmaNativeRunner.ENABLE_EXPERIMENTAL_RAW_LITERTLM_PROPERTY);
        System.clearProperty(LiteRTGemmaNativeRunner.ALLOW_RAW_LITERTLM_GPU_ON_MACOS_PROPERTY);
        System.clearProperty(LiteRTGemmaNativeRunner.ENABLE_RAW_LITERTLM_AUX_GPU_ON_MACOS_PROPERTY);
        System.clearProperty(LiteRTGemmaNativeRunner.ENABLE_STATEFUL_KV_BUFFERS_PROPERTY);
        System.clearProperty(LiteRTLmPythonBridge.ENABLE_PROPERTY);
        System.clearProperty(LiteRTLmPythonBridge.BACKEND_PROPERTY);
        System.clearProperty(LiteRTLmPythonBridge.TIMEOUT_SECONDS_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.ENABLE_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.BACKEND_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.TIMEOUT_SECONDS_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.MAX_NUM_TOKENS_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.NORMALIZE_SHORT_QUESTIONS_PROPERTY);
        System.clearProperty(LiteRTLmJvmBridge.ENABLE_SPECULATIVE_DECODING_PROPERTY);
    }

    @Test
    void rawLiteRtLmExperimentalPathIsOptIn() {
        assertFalse(LiteRTGemmaNativeRunner.experimentalRawLiteRtLmEnabled());

        System.setProperty(LiteRTGemmaNativeRunner.ENABLE_EXPERIMENTAL_RAW_LITERTLM_PROPERTY, "true");

        assertTrue(LiteRTGemmaNativeRunner.experimentalRawLiteRtLmEnabled());
    }

    @Test
    void rawLiteRtLmGpuOnMacOsProbeIsOptIn() {
        assertFalse(LiteRTGemmaNativeRunner.rawLiteRtLmGpuOnMacOsAllowed());

        System.setProperty(LiteRTGemmaNativeRunner.ALLOW_RAW_LITERTLM_GPU_ON_MACOS_PROPERTY, "true");

        assertTrue(LiteRTGemmaNativeRunner.rawLiteRtLmGpuOnMacOsAllowed());
    }

    @Test
    void rawLiteRtLmAuxGpuOnMacOsProbeIsOptIn() {
        assertFalse(LiteRTGemmaNativeRunner.rawLiteRtLmAuxGpuOnMacOsEnabled());

        System.setProperty(LiteRTGemmaNativeRunner.ENABLE_RAW_LITERTLM_AUX_GPU_ON_MACOS_PROPERTY, "true");

        assertTrue(LiteRTGemmaNativeRunner.rawLiteRtLmAuxGpuOnMacOsEnabled());
    }

    @Test
    void statefulKvBuffersAreEnabledByDefaultButCanBeDisabled() {
        assertTrue(LiteRTGemmaNativeRunner.statefulKvBuffersEnabled());

        System.setProperty(LiteRTGemmaNativeRunner.ENABLE_STATEFUL_KV_BUFFERS_PROPERTY, "false");

        assertFalse(LiteRTGemmaNativeRunner.statefulKvBuffersEnabled());

        System.setProperty(LiteRTGemmaNativeRunner.ENABLE_STATEFUL_KV_BUFFERS_PROPERTY, "true");
        assertTrue(LiteRTGemmaNativeRunner.statefulKvBuffersEnabled());
    }

    @Test
    void officialLiteRtLmBridgeIsEnabledByDefaultButCanBeDisabled() {
        assertTrue(LiteRTLmPythonBridge.enabled());

        System.setProperty(LiteRTLmPythonBridge.ENABLE_PROPERTY, "false");

        assertFalse(LiteRTLmPythonBridge.enabled());
    }

    @Test
    void officialLiteRtLmBridgeBackendDefaultsToGpuWhenRequested() {
        assertEquals("GPU", LiteRTLmPythonBridge.backendName(true));
        assertEquals("CPU", LiteRTLmPythonBridge.backendName(false));

        System.setProperty(LiteRTLmPythonBridge.BACKEND_PROPERTY, "metal");

        assertEquals("GPU", LiteRTLmPythonBridge.backendName(false));
    }

    @Test
    void officialLiteRtLmJvmBridgeIsEnabledByDefaultButCanBeDisabled() {
        assertTrue(LiteRTLmJvmBridge.enabled());

        System.setProperty(LiteRTLmJvmBridge.ENABLE_PROPERTY, "false");

        assertFalse(LiteRTLmJvmBridge.enabled());
    }

    @Test
    void officialLiteRtLmJvmBridgeBackendMapsMetalToGpu() {
        assertEquals("GPU", LiteRTLmJvmBridge.backendName(true));
        assertEquals("CPU", LiteRTLmJvmBridge.backendName(false));

        System.setProperty(LiteRTLmJvmBridge.BACKEND_PROPERTY, "metal");

        assertEquals("GPU", LiteRTLmJvmBridge.backendName(false));
    }

    @Test
    void officialLiteRtLmJvmBridgeNormalizesBareQuestionsForChatQuality() {
        assertEquals(
                "Answer directly and concisely in one or two sentences. Do not ask a clarification question. "
                        + "If a term is ambiguous, answer the most common meaning first and mention common alternatives briefly.\n"
                        + "Question: What is jakarta?",
                LiteRTLmJvmBridge.promptForModel("what is jakarta"));
    }

    @Test
    void officialLiteRtLmJvmBridgePromptNormalizationCanBeDisabled() {
        System.setProperty(LiteRTLmJvmBridge.NORMALIZE_SHORT_QUESTIONS_PROPERTY, "false");

        assertEquals("what is jakarta", LiteRTLmJvmBridge.promptForModel("what is jakarta"));
    }

    @Test
    void officialLiteRtLmJvmBridgeEnablesMtpForGemma4GpuByDefault() {
        assertEquals(Boolean.TRUE, LiteRTLmJvmBridge.speculativeDecodingPreference(
                true, Path.of("gemma-4-E2B-it.litertlm")));
        assertEquals(null, LiteRTLmJvmBridge.speculativeDecodingPreference(
                false, Path.of("gemma-4-E2B-it.litertlm")));
        assertEquals(null, LiteRTLmJvmBridge.speculativeDecodingPreference(
                true, Path.of("gemma3-1b-it.litertlm")));
    }

    @Test
    void officialLiteRtLmJvmBridgeMtpCanBeOverridden() {
        System.setProperty(LiteRTLmJvmBridge.ENABLE_SPECULATIVE_DECODING_PROPERTY, "false");
        assertEquals(Boolean.FALSE, LiteRTLmJvmBridge.speculativeDecodingPreference(
                true, Path.of("gemma-4-E2B-it.litertlm")));

        System.setProperty(LiteRTLmJvmBridge.ENABLE_SPECULATIVE_DECODING_PROPERTY, "true");
        assertEquals(Boolean.TRUE, LiteRTLmJvmBridge.speculativeDecodingPreference(
                false, Path.of("gemma3-1b-it.litertlm")));
    }

    @Test
    void officialLiteRtLmJvmBridgeBoundsGemma4KvCacheByDefault() {
        assertEquals(2048, LiteRTLmJvmBridge.maxNumTokens(Path.of("gemma-4-E2B-it.litertlm")));
        assertEquals(null, LiteRTLmJvmBridge.maxNumTokens(Path.of("gemma3-1b-it.litertlm")));
    }

    @Test
    void officialLiteRtLmJvmBridgeMaxNumTokensCanUseModelDefault() {
        System.setProperty(LiteRTLmJvmBridge.MAX_NUM_TOKENS_PROPERTY, "model");

        assertEquals(null, LiteRTLmJvmBridge.maxNumTokens(Path.of("gemma-4-E2B-it.litertlm")));
    }

    @Test
    void officialLiteRtLmJvmBridgeMaxNumTokensCanBeOverridden() {
        System.setProperty(LiteRTLmJvmBridge.MAX_NUM_TOKENS_PROPERTY, "4096");

        assertEquals(4096, LiteRTLmJvmBridge.maxNumTokens(Path.of("gemma-4-E2B-it.litertlm")));
    }

    @Test
    void officialLiteRtLmJvmBridgeApproximateLimiterAvoidsRescanningChunks() {
        LiteRTLmJvmBridge.ApproximateTokenStreamLimiter limiter =
                new LiteRTLmJvmBridge.ApproximateTokenStreamLimiter(4);

        String output = limiter.offer("Indonesia is ")
                + limiter.offer("in Southeast ")
                + limiter.offer("Asia with many islands");

        assertEquals("Indonesia is in Southeast", output);
        assertEquals(4, limiter.emittedTokenCount());
        assertTrue(limiter.atLimit());
    }

    @Test
    void plainWherePromptIsRewrittenAsLocationQuestion() {
        assertEquals("Question: What country, region, or place is Jakarta located in?\\nAnswer:",
                LiteRTGemmaNativeRunner.formatPlainPrompt("where is jakarta"));
    }

    @Test
    void nonWherePlainPromptStillUsesQaCue() {
        assertEquals("Question: what is jakarta\\nAnswer:",
                LiteRTGemmaNativeRunner.formatPlainPrompt("what is jakarta"));
    }

    @Test
    void litertLmFilePrefersRunnableTaskSibling(@TempDir Path tempDir) throws Exception {
        Path nativeFile = tempDir.resolve("gemma-4-E2B-it.litertlm");
        Path taskFile = tempDir.resolve("gemma-4-E2B-it-web.task");
        Files.write(nativeFile, new byte[]{1});
        Files.write(taskFile, new byte[]{2});

        Optional<Path> resolved = LiteRTContainerParser.findBestModelFile(nativeFile);

        assertTrue(resolved.isPresent());
        assertEquals(taskFile, resolved.get());
    }

    @Test
    void defaultThresholdSkipsShortPrompts() {
        assertEquals(4, LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens());
        assertFalse(LiteRTGemmaNativeRunner.shouldUseRawPrefill(0));
        assertFalse(LiteRTGemmaNativeRunner.shouldUseRawPrefill(3));
    }

    @Test
    void defaultThresholdAllowsLongerPrompts() {
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(4));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(8));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(64));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(128));
    }

    @Test
    void overrideCanLowerRawPrefillThreshold() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_MIN_TOKENS_PROPERTY, "8");

        assertEquals(8, LiteRTGemmaNativeRunner.rawPrefillMinTokens());
        assertFalse(LiteRTGemmaNativeRunner.shouldUseRawPrefill(7));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(8));
    }

    @Test
    void zeroThresholdAllowsShortPromptsToUseRawPrefill() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_MIN_TOKENS_PROPERTY, "0");

        assertEquals(0, LiteRTGemmaNativeRunner.rawPrefillMinTokens());
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(0));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(1));
    }

    @Test
    void negativeThresholdFallsBackToDefault() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_MIN_TOKENS_PROPERTY, "-1");

        assertEquals(LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens(),
                LiteRTGemmaNativeRunner.rawPrefillMinTokens());
        assertFalse(LiteRTGemmaNativeRunner.shouldUseRawPrefill(
                LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens() - 1));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(
                LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens()));
    }

    @Test
    void invalidThresholdFallsBackToDefault() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_MIN_TOKENS_PROPERTY, "abc");

        assertEquals(LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens(),
                LiteRTGemmaNativeRunner.rawPrefillMinTokens());
        assertFalse(LiteRTGemmaNativeRunner.shouldUseRawPrefill(
                LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens() - 1));
        assertTrue(LiteRTGemmaNativeRunner.shouldUseRawPrefill(
                LiteRTGemmaNativeRunner.defaultRawPrefillMinTokens()));
    }

    @Test
    void defaultDecodeTailTokensAreReserved() {
        assertEquals(0, LiteRTGemmaNativeRunner.defaultRawPrefillDecodeTailTokens());
        assertEquals(0, LiteRTGemmaNativeRunner.rawPrefillDecodeTailTokens());
    }

    @Test
    void decodeTailOverrideCanBeLowered() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY, "4");

        assertEquals(4, LiteRTGemmaNativeRunner.rawPrefillDecodeTailTokens());
    }

    @Test
    void negativeDecodeTailFallsBackToDefault() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY, "-1");

        assertEquals(LiteRTGemmaNativeRunner.defaultRawPrefillDecodeTailTokens(),
                LiteRTGemmaNativeRunner.rawPrefillDecodeTailTokens());
    }

    @Test
    void invalidDecodeTailFallsBackToDefault() {
        System.setProperty(LiteRTGemmaNativeRunner.RAW_PREFILL_DECODE_TAIL_TOKENS_PROPERTY, "abc");

        assertEquals(LiteRTGemmaNativeRunner.defaultRawPrefillDecodeTailTokens(),
                LiteRTGemmaNativeRunner.rawPrefillDecodeTailTokens());
    }
}
