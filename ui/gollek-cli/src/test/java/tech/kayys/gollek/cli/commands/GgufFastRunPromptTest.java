package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.alkhawarizm.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufFastRunPromptTest {

    @Test
    void runHeaderMatchesFullCliShapeForIndexedGgufModels() {
        GgufFastRun.FastArgs args = GgufFastRun.FastArgs.parse(
                new String[] {"run", "--model", "b71c9d", "--prompt", "hello"});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        GgufFastRun.printRunHeader(
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"),
                args,
                new PrintStream(bytes, true, StandardCharsets.UTF_8));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("_____"));
        assertTrue(output.contains("Resolved local model index entry: /tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));
        assertTrue(output.contains("Model: /tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));
        assertTrue(output.contains("Provider: gguf, format=gguf"));
        assertFalse(output.contains("Execution route:"));
    }

    @Test
    void runHeaderDoesNotPrintResolvedLineForExplicitGgufModelFile() {
        GgufFastRun.FastArgs args = GgufFastRun.FastArgs.parse(
                new String[] {"run", "--model-file", "/tmp/model.gguf", "--prompt", "hello"});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        GgufFastRun.printRunHeader(
                Path.of("/tmp/model.gguf"),
                args,
                new PrintStream(bytes, true, StandardCharsets.UTF_8));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertFalse(output.contains("Resolved local model index entry:"));
        assertTrue(output.contains("Model: /tmp/model.gguf"));
        assertTrue(output.contains("Provider: gguf, format=gguf"));
    }

    @Test
    void runHeaderCanSuppressBannerForFullCliHandoffs() {
        GgufFastRun.FastArgs args = GgufFastRun.FastArgs.parse(
                new String[] {"run", "--no-banner", "--model", "b71c9d", "--prompt", "hello"});
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        GgufFastRun.printRunHeader(
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"),
                args,
                new PrintStream(bytes, true, StandardCharsets.UTF_8));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertFalse(output.contains("_____"));
        assertTrue(output.contains("Resolved local model index entry: /tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));
        assertTrue(output.contains("Model: /tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));
        assertTrue(output.contains("Provider: gguf, format=gguf"));
    }

    @Test
    void executionRouteNamesGgufDaemonAndDirectFastPath() {
        assertEquals(
                "Execution route: gguf (backend=MTL0 (Apple M4)) [llama_cpp_gguf_daemon]",
                GgufFastRun.executionRouteLine("MTL0 (Apple M4)", "daemon"));
        assertEquals(
                "Execution route: gguf (backend=CPU) [llama_cpp_gguf_fast_path]",
                GgufFastRun.executionRouteLine("CPU", "fast path"));
    }

    @Test
    void sessionReuseLabelsExposeColdExactAndCompatibleDaemonHits() {
        assertEquals("one-shot", GgufFastRun.sessionReuseLabel(false, false, "cold"));
        assertEquals("cold", GgufFastRun.sessionReuseLabel(true, false, "cold"));
        assertEquals("warm-exact", GgufFastRun.sessionReuseLabel(true, true, "exact"));
        assertEquals("warm-compatible", GgufFastRun.sessionReuseLabel(true, true, "compatible"));
    }

    @Test
    void sessionCoverageSummaryShowsWarmShapeReuseBounds() {
        assertEquals("ctx<=2048,batch<=1024,ubatch<=512",
                GgufFastRun.sessionCoverageSummary(2048, 1024, 512));
    }

    @Test
    void outputBufferSizingUsesSafetyFloorAndTokenScaledCapacity() {
        assertEquals(64L * 1024L, GgufFastRun.outputBufferBytesForTokens(1));
        assertEquals(128L * 1024L, GgufFastRun.outputBufferBytesForTokens(256));
    }

    @Test
    void outputBufferRetryDoublesCapacityAndDetectsNativeOverflow() {
        assertEquals(128L * 1024L, GgufFastRun.nextOutputBufferBytes(64L * 1024L));
        assertTrue(GgufFastRun.outputBufferTooSmall(-9, ""));
        assertTrue(GgufFastRun.outputBufferTooSmall(-7, "output buffer too small"));
        assertFalse(GgufFastRun.outputBufferTooSmall(-7, "token conversion failed"));
    }

    @Test
    void fastRunStatsUseCliPerformanceMetricShape() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        long start = TimeUnit.MILLISECONDS.toNanos(1_000);
        long open = TimeUnit.MILLISECONDS.toNanos(125);
        long generate = TimeUnit.MILLISECONDS.toNanos(500);
        long end = start + open + generate;

        GgufFastRun.printFastRunStats(
                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                "Fast GGUF",
                10,
                start,
                open,
                generate,
                end);

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("[Fast GGUF, Duration:"));
        assertTrue(output.contains("Speed:"));
        assertTrue(output.contains("Performance Metrics:"));
        assertTrue(output.contains("  open time      ="));
        assertTrue(output.contains("  generate call  ="));
        assertTrue(output.contains("  generation     ="));
        assertTrue(output.contains("10 tokens"));
        assertTrue(output.contains("  token latency  ="));
    }

    @Test
    void fastRunStatsIncludeNativeBackendAndTimingBreakdownWhenAvailable() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        long start = TimeUnit.MILLISECONDS.toNanos(1_000);
        long open = TimeUnit.MILLISECONDS.toNanos(25);
        long generate = TimeUnit.MILLISECONDS.toNanos(100);
        long end = start + open + generate;
        String nativeMetrics = "backend=Metal (Apple M4), gpuLayers=-1, threads=8, ctx=512, batch=512, "
                + "modelLoad=1.500ms, contextInit=2.500ms, tokenize=0.200ms, prefill=12.000ms, decode=34.000ms, "
                + "tokenizeCache=hit, tokenCacheTokens=9, "
                + "sampler=reused, samplerMs=0.010ms, "
                + "promptCache=below-threshold, promptCacheMs=0.000ms, promptTokens=9, "
                + "generatedTokens=10, decodedTokens=9, repeatedPrompt=true, "
                + "promptCacheEagerShort=true, outputBytes=42, "
                + "warmSession=true, sessionReuse=exact, "
                + "javaOutputBufferBytes=65536, javaOutputRetries=1";

        GgufFastRun.printFastRunStats(
                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                "Fast GGUF",
                10,
                start,
                open,
                generate,
                end,
                Optional.of(nativeMetrics));

        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("  native backend = Metal (Apple M4)"));
        assertTrue(output.contains("gpuLayers=-1"));
        assertTrue(output.contains("context init ="));
        assertTrue(output.contains("  native tokenize="));
        assertTrue(output.contains("(hit)"));
        assertTrue(output.contains("(9 tokens)"));
        assertTrue(output.contains("(10 sampled, 9 decoded)"));
        assertTrue(output.contains("  native sampler = reused"));
        assertTrue(output.contains("  native session = warm (exact)"));
        assertTrue(output.contains("  native cache   = below-threshold, 0.00 ms (eager-short, repeat)")
                || output.contains("  native cache   = below-threshold, 0,00 ms (eager-short, repeat)"),
                output);
        assertTrue(output.contains("  native output  = 42 bytes, java buffer = 65536 bytes, retries = 1"));
    }

    @Test
    void parsesNativeMetricStringIntoStableKeyValuePairs() {
        Map<String, String> metrics = GgufFastRun.parseNativeMetrics(
                "backend=Metal (Apple M4), gpuLayers=-1, prefill=12.000ms, promptCache=hit, "
                        + "tokenizeCache=hit, sampler=reused, samplerMs=0.010ms, "
                        + "promptCacheEagerShort=true, repeatedPrompt=true, "
                        + "warmSession=true, sessionReuse=compatible, "
                        + "javaOutputRetries=1");

        assertEquals("Metal (Apple M4)", metrics.get("backend"));
        assertEquals("-1", metrics.get("gpuLayers"));
        assertEquals("12.000ms", metrics.get("prefill"));
        assertEquals("hit", metrics.get("promptCache"));
        assertEquals("hit", metrics.get("tokenizeCache"));
        assertEquals("reused", metrics.get("sampler"));
        assertEquals("true", metrics.get("promptCacheEagerShort"));
        assertEquals("true", metrics.get("repeatedPrompt"));
        assertEquals("true", metrics.get("warmSession"));
        assertEquals("compatible", metrics.get("sessionReuse"));
        assertEquals("1", metrics.get("javaOutputRetries"));
    }

    @Test
    void wrapsGemma4PromptWithTurnTokens() {
        String formatted = GgufFastRun.formatPromptForModel(
                "where is jakarta",
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

        assertEquals("<|turn>user\nAnswer directly and concisely.\nQuestion: where is jakarta<turn|>\n<|turn>model\n", formatted);
    }

    @Test
    void keepsGemma4NonQuestionPromptWithoutConciseQuestionInstruction() {
        String formatted = GgufFastRun.formatPromptForModel(
                "write a tiny poem about jakarta",
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

        assertEquals("<|turn>user\nwrite a tiny poem about jakarta<turn|>\n<|turn>model\n", formatted);
    }

    @Test
    void conciseQuestionInstructionCanBeDisabled() {
        String previous = System.getProperty("gollek.gguf.fast_run.concise_qa_prompt");
        System.setProperty("gollek.gguf.fast_run.concise_qa_prompt", "false");
        try {
            String formatted = GgufFastRun.formatPromptForModel(
                    "where is jakarta",
                    Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

            assertEquals("<|turn>user\nwhere is jakarta<turn|>\n<|turn>model\n", formatted);
        } finally {
            restoreProperty("gollek.gguf.fast_run.concise_qa_prompt", previous);
        }
    }

    @Test
    void keepsPreformattedPromptUntouched() {
        String prompt = "<|turn>user\nwhere is jakarta<turn|>\n<|turn>model\n";

        assertEquals(prompt, GgufFastRun.formatPromptForModel(
                prompt,
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf")));
    }

    @Test
    void keepsNonGemma4PromptRawInAutoMode() {
        assertEquals("where is jakarta", GgufFastRun.formatPromptForModel(
                "where is jakarta",
                Path.of("/tmp/llama.gguf")));
    }

    @Test
    void providerJavaSelectsJavaEngineByDefault() {
        assertEquals("JAVA", GgufFastRun.effectiveEngineModeName("java", "auto"));
    }

    @Test
    void explicitEngineOverridesProviderDefault() {
        assertEquals("LLAMA_CPP", GgufFastRun.effectiveEngineModeName("java", "llama.cpp"));
    }

    @Test
    void benchmarkAliasSelectsBenchmarkMode() {
        assertEquals("BENCHMARK", GgufFastRun.effectiveEngineModeName("gguf", "compare"));
    }

    @Test
    void javaReadinessSummaryIsMachineReadable() {
        GgufRuntimeProfile profile = new GgufRuntimeProfile(
                "gemma4",
                3,
                1,
                1,
                1024,
                1024,
                1024,
                2,
                2,
                0,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                new ModelConfig(),
                1);

        assertEquals(
                "loaderReady=true, decoderTensorsReady=true, rowDotReady=true, generationReady=false",
                GgufFastRun.javaReadinessSummary(profile));
    }

    @Test
    void boundsBatchConfigurationToContextWindow() {
        String previous = System.getProperty("gollek.gguf.fast_run.batch");
        System.setProperty("gollek.gguf.fast_run.batch", "4096");
        try {
            assertEquals(2048, GgufFastRun.boundedBatch("gollek.gguf.fast_run.batch", 1024, 2048));
        } finally {
            restoreProperty("gollek.gguf.fast_run.batch", previous);
        }
    }

    @Test
    void keepsBatchConfigurationPositive() {
        String previous = System.getProperty("gollek.gguf.fast_run.ubatch");
        System.setProperty("gollek.gguf.fast_run.ubatch", "0");
        try {
            assertEquals(1, GgufFastRun.boundedBatch("gollek.gguf.fast_run.ubatch", 512, 1024));
        } finally {
            restoreProperty("gollek.gguf.fast_run.ubatch", previous);
        }
    }

    @Test
    void fastRunConfigEnvNamesIncludeShellFriendlyAliases() {
        assertTrue(GgufFastRun.configEnvNames("gollek.gguf.fast_run.gpu_layers")
                .contains("GOLLEK_GGUF_FAST_GPU_LAYERS"));
        assertTrue(GgufFastRun.configEnvNames("gollek.gguf.fast_run.gpu_layers")
                .contains("GGUF_GPU_LAYERS"));
        assertTrue(GgufFastRun.configEnvNames("gollek.gguf.fast_run.batch")
                .contains("GGUF_BATCH_SIZE"));
        assertTrue(GgufFastRun.configEnvNames("gollek.gguf.fast_run.context")
                .contains("GGUF_CONTEXT_SIZE"));
        assertTrue(GgufFastRun.configEnvNames("gollek.gguf.fast_run.daemon")
                .contains("GOLLEK_GGUF_FAST_DAEMON"));
    }

    @Test
    void autoContextUsesSmallWindowForShortPrompt() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.clearProperty("gollek.gguf.fast_run.context");
        try {
            assertEquals(512, GgufFastRun.fastRunContext("where is jakarta", 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void autoContextGrowsForLongPrompt() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.clearProperty("gollek.gguf.fast_run.context");
        try {
            assertEquals(2048, GgufFastRun.fastRunContext("x".repeat(1000), 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void explicitContextOverridesAutoSizing() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.setProperty("gollek.gguf.fast_run.context", "1536");
        try {
            assertEquals(1536, GgufFastRun.fastRunContext("where is jakarta", 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void prewarmContextDefaultsToReusableMaxAutoContext() {
        String previousContext = System.getProperty("gollek.gguf.fast_run.context");
        String previousMaxAuto = System.getProperty("gollek.gguf.fast_run.max_auto_context");
        String previousPrewarmMin = System.getProperty("gollek.gguf.fast_run.prewarm_min_context");
        System.clearProperty("gollek.gguf.fast_run.context");
        System.clearProperty("gollek.gguf.fast_run.max_auto_context");
        System.clearProperty("gollek.gguf.fast_run.prewarm_min_context");
        try {
            assertEquals(2048, GgufFastRun.fastRunPrewarmContext("where is jakarta", 10));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previousContext);
            restoreProperty("gollek.gguf.fast_run.max_auto_context", previousMaxAuto);
            restoreProperty("gollek.gguf.fast_run.prewarm_min_context", previousPrewarmMin);
        }
    }

    @Test
    void prewarmContextHonorsExplicitContextOverride() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.setProperty("gollek.gguf.fast_run.context", "1536");
        try {
            assertEquals(1536, GgufFastRun.fastRunPrewarmContext("where is jakarta", 10));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void prewarmMinContextCanBeLoweredForMemoryConstrainedRuns() {
        String previousContext = System.getProperty("gollek.gguf.fast_run.context");
        String previousMaxAuto = System.getProperty("gollek.gguf.fast_run.max_auto_context");
        String previousPrewarmMin = System.getProperty("gollek.gguf.fast_run.prewarm_min_context");
        System.clearProperty("gollek.gguf.fast_run.context");
        System.setProperty("gollek.gguf.fast_run.max_auto_context", "2048");
        System.setProperty("gollek.gguf.fast_run.prewarm_min_context", "512");
        try {
            assertEquals(512, GgufFastRun.fastRunPrewarmContext("where is jakarta", 10));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previousContext);
            restoreProperty("gollek.gguf.fast_run.max_auto_context", previousMaxAuto);
            restoreProperty("gollek.gguf.fast_run.prewarm_min_context", previousPrewarmMin);
        }
    }

    @Test
    void fastRunSwaFullDefaultsToLlamaCppCommonMode() {
        String previous = System.getProperty("gollek.gguf.fast_run.swa_full");
        System.clearProperty("gollek.gguf.fast_run.swa_full");
        try {
            assertFalse(GgufFastRun.fastRunSwaFull());
        } finally {
            restoreProperty("gollek.gguf.fast_run.swa_full", previous);
        }
    }

    @Test
    void fastRunSwaFullCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.swa_full");
        System.setProperty("gollek.gguf.fast_run.swa_full", "true");
        try {
            assertTrue(GgufFastRun.fastRunSwaFull());
        } finally {
            restoreProperty("gollek.gguf.fast_run.swa_full", previous);
        }
    }

    @Test
    void fastRunBooleanConfigAcceptsShellStyleTruthValues() {
        String previous = System.getProperty("gollek.gguf.fast_run.swa_full");
        System.setProperty("gollek.gguf.fast_run.swa_full", "yes");
        try {
            assertTrue(GgufFastRun.fastRunSwaFull());
        } finally {
            restoreProperty("gollek.gguf.fast_run.swa_full", previous);
        }
    }

    @Test
    void fastRunCpuFallbackIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.cpu_fallback");
        System.clearProperty("gollek.gguf.fast_run.cpu_fallback");
        try {
            assertTrue(GgufFastRun.fastRunCpuFallback());
        } finally {
            restoreProperty("gollek.gguf.fast_run.cpu_fallback", previous);
        }
    }

    @Test
    void fastRunCpuFallbackCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.cpu_fallback");
        System.setProperty("gollek.gguf.fast_run.cpu_fallback", "false");
        try {
            assertFalse(GgufFastRun.fastRunCpuFallback());
        } finally {
            restoreProperty("gollek.gguf.fast_run.cpu_fallback", previous);
        }
    }

    @Test
    void fastRunTimingIsDisabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.timing");
        System.clearProperty("gollek.gguf.fast_run.timing");
        try {
            assertFalse(GgufFastRun.fastRunTiming());
        } finally {
            restoreProperty("gollek.gguf.fast_run.timing", previous);
        }
    }

    @Test
    void fastRunTimingCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.timing");
        System.setProperty("gollek.gguf.fast_run.timing", "true");
        try {
            assertTrue(GgufFastRun.fastRunTiming());
        } finally {
            restoreProperty("gollek.gguf.fast_run.timing", previous);
        }
    }

    @Test
    void oneShotNativeRunUsesHardExitByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertTrue(GgufFastRun.shouldHardExitAfterNativeRun(false));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void daemonNativeRunDoesNotHardExitAfterRequest() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertFalse(GgufFastRun.shouldHardExitAfterNativeRun(true));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void oneShotNativeHardExitCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.setProperty("gollek.gguf.fast_run.hard_exit_after_run", "false");
        try {
            assertFalse(GgufFastRun.shouldHardExitAfterNativeRun(false));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void staleDaemonForceKillIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        System.clearProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        try {
            assertTrue(GgufFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonForceKillCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        System.setProperty("gollek.gguf.fast_run.stale_daemon_force_kill", "false");
        try {
            assertFalse(GgufFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonKillWaitHasSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms");
        System.setProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms", "1");
        try {
            assertEquals(100L, GgufFastRun.staleDaemonKillWaitMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms", previous);
        }
    }

    @Test
    void daemonLauncherUsesPlatformDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.clearProperty("gollek.gguf.fast_run.daemon_launcher");
        try {
            assertEquals(GgufFastRun.defaultDaemonLauncherMode(), GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsAutoOverride() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", " auto ");
        try {
            assertEquals("auto", GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsLaunchctlOverride() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", " launchctl ");
        try {
            assertEquals("launchctl", GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherFallsBackToNohupWhenInvalid() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", "bogus");
        try {
            assertEquals(GgufFastRun.defaultDaemonLauncherMode(), GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonRequestRetryDefaultsToShortBoundedWindow() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        System.clearProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        try {
            assertEquals(2_000L, GgufFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetryCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        System.setProperty("gollek.gguf.fast_run.daemon_request_retry_ms", "-1");
        try {
            assertEquals(0L, GgufFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetrySleepHasSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms");
        System.setProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms", "1");
        try {
            assertEquals(10L, GgufFastRun.daemonRequestRetrySleepMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms", previous);
        }
    }

    @Test
    void daemonMaxSessionsDefaultsToSmallBoundedCache() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_max_sessions");
        System.clearProperty("gollek.gguf.fast_run.daemon_max_sessions");
        try {
            assertEquals(2, GgufFastRun.daemonMaxSessions());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_max_sessions", previous);
        }
    }

    @Test
    void daemonMaxSessionsHasSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_max_sessions");
        System.setProperty("gollek.gguf.fast_run.daemon_max_sessions", "0");
        try {
            assertEquals(1, GgufFastRun.daemonMaxSessions());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_max_sessions", previous);
        }
    }

    @Test
    void prewarmArgsAddFastPathDefaults() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {"--model", "b71c9d"}));
    }

    @Test
    void prewarmArgsPreserveExplicitPromptAndContextTokens() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {
                        "--model", "b71c9d",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64"
                }));
    }

    @Test
    void prewarmArgsStripPublicCommandName() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {"prewarm", "--model", "b71c9d"}));
    }

    @Test
    void prewarmReturnsFallbackForLiteRtProvider() {
        assertEquals(42, GgufFastRun.prewarm(new String[] {
                "--provider", "litert",
                "--model", "7c51c9"
        }));
    }

    @Test
    void prewarmReturnsFallbackForLiteRtModelFileWithoutProvider() {
        assertEquals(42, GgufFastRun.prewarm(new String[] {
                "--model-file", "/tmp/gemma-4-E2B-it.litertlm"
        }));
    }

    @Test
    void runReturnsFallbackForLiteRtModelFileWithoutProvider() {
        assertEquals(42, GgufFastRun.run(new String[] {
                "run",
                "--model-file", "/tmp/gemma-4-E2B-it.litertlm",
                "--prompt", "where is jakarta"
        }));
    }

    @Test
    void safetensorShortIdCanResolveMatchingGgufEquivalentForSpeed(@TempDir Path tempHome) throws Exception {
        Path ggufDir = tempHome.resolve("models/gguf/gemma");
        Path ggufFile = ggufDir.resolve("google__gemma-4-E2B-it-Q4_K_M.gguf");
        Path safetensorDir = tempHome.resolve("models/safetensors/gemma");
        Files.createDirectories(ggufDir);
        Files.createDirectories(safetensorDir);
        Files.writeString(ggufFile, "stub");
        writeModelIndex(tempHome, safetensorDir, ggufFile);

        String previousHome = System.getProperty("gollek.home");
        String previousAutoEquivalent = System.getProperty("gollek.gguf.fast_run.auto_equivalent");
        System.setProperty("gollek.home", tempHome.toString());
        System.clearProperty("gollek.gguf.fast_run.auto_equivalent");
        try {
            GgufFastRun.FastArgs args = GgufFastRun.FastArgs.parse(
                    new String[] {"run", "--model", "97cbf2", "--prompt", "hello"});

            assertEquals(ggufFile.toAbsolutePath().normalize(), GgufFastRun.resolveGgufModel(args).orElseThrow());
        } finally {
            restoreProperty("gollek.home", previousHome);
            restoreProperty("gollek.gguf.fast_run.auto_equivalent", previousAutoEquivalent);
        }
    }

    @Test
    void equivalentGgufResolutionCanBeDisabled(@TempDir Path tempHome) throws Exception {
        Path ggufDir = tempHome.resolve("models/gguf/gemma");
        Path ggufFile = ggufDir.resolve("google__gemma-4-E2B-it-Q4_K_M.gguf");
        Path safetensorDir = tempHome.resolve("models/safetensors/gemma");
        Files.createDirectories(ggufDir);
        Files.createDirectories(safetensorDir);
        Files.writeString(ggufFile, "stub");
        writeModelIndex(tempHome, safetensorDir, ggufFile);

        String previousHome = System.getProperty("gollek.home");
        String previousAutoEquivalent = System.getProperty("gollek.gguf.fast_run.auto_equivalent");
        System.setProperty("gollek.home", tempHome.toString());
        System.setProperty("gollek.gguf.fast_run.auto_equivalent", "false");
        try {
            GgufFastRun.FastArgs args = GgufFastRun.FastArgs.parse(
                    new String[] {"run", "--model", "97cbf2", "--prompt", "hello"});

            assertTrue(GgufFastRun.resolveGgufModel(args).isEmpty());
        } finally {
            restoreProperty("gollek.home", previousHome);
            restoreProperty("gollek.gguf.fast_run.auto_equivalent", previousAutoEquivalent);
        }
    }

    @Test
    void prewarmTokenCountDefaultsToTwoSoDecodePathWarms() {
        String previous = System.getProperty("gollek.gguf.fast_run.prewarm_tokens");
        System.clearProperty("gollek.gguf.fast_run.prewarm_tokens");
        try {
            assertEquals(2, GgufFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.gguf.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void prewarmTokenCountKeepsDecodeWarmupSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.prewarm_tokens");
        System.setProperty("gollek.gguf.fast_run.prewarm_tokens", "1");
        try {
            assertEquals(2, GgufFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.gguf.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void daemonKeyMatchingIsRelaxedByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.strict_daemon_key");
        System.clearProperty("gollek.gguf.fast_run.strict_daemon_key");
        try {
            assertFalse(GgufFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.gguf.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void strictDaemonKeyCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.strict_daemon_key");
        System.setProperty("gollek.gguf.fast_run.strict_daemon_key", "true");
        try {
            assertTrue(GgufFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.gguf.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void hardExitAfterRunIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertTrue(GgufFastRun.hardExitAfterRun());
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void runCommandRoutesExplicitGgufProviderToStandaloneFastPath() {
        RunCommand command = new RunCommand();
        command.modelId = "b71c9d";
        command.prompt = "where is jakarta";
        command.providerId = "gguf";
        command.ggufEngine = "java";
        command.ggufBackend = "cpu";
        command.maxTokens = 10;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--model", "b71c9d",
                "--prompt", "where is jakarta",
                "--max-tokens", "10",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--engine", "java",
                "--backend", "cpu",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    @Test
    void runCommandCanSuppressDelegatedGgufBannerAfterFullCliHeader() {
        RunCommand command = new RunCommand();
        command.modelId = "b71c9d";
        command.prompt = "where is jakarta";
        command.providerId = "gguf";
        command.maxTokens = 1;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertArrayEquals(new String[] {
                "run",
                "--no-banner",
                "--model", "b71c9d",
                "--prompt", "where is jakarta",
                "--max-tokens", "1",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs(true));
    }

    @Test
    void runCommandDoesNotStealGgufFileFromExplicitNonGgufProvider() {
        RunCommand command = new RunCommand();
        command.modelFile = "/tmp/model.gguf";
        command.prompt = "where is jakarta";
        command.providerId = "litert";

        assertFalse(command.shouldTryStandaloneGgufFastPath());
    }

    @Test
    void runCommandRoutesGgufFileWithoutProviderToStandaloneFastPath() {
        RunCommand command = new RunCommand();
        command.modelFile = "/tmp/model.gguf";
        command.prompt = "where is jakarta";
        command.maxTokens = 1;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--modelFile", "/tmp/model.gguf",
                "--prompt", "where is jakarta",
                "--max-tokens", "1",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    @Test
    void runCommandRouteReportModeDoesNotEnterStandaloneGgufFastPath() {
        RunCommand command = new RunCommand();
        command.modelFile = "/tmp/model.gguf";
        command.prompt = "where is jakarta";
        command.routeReportJson = true;

        assertFalse(command.shouldTryStandaloneGgufFastPath());
    }

    @Test
    void routeReportModeUsesLocalOnlyRepositoryResolutionByDefault() {
        RunCommand command = new RunCommand();
        command.routeReportJson = true;

        assertFalse(command.shouldAllowRepositoryResolutionDuringRouteReport());

        command.routeReportAllowPull = true;

        assertTrue(command.shouldAllowRepositoryResolutionDuringRouteReport());
        assertEquals(0, RoutePreflightReport.exitCode("/tmp/model.gguf", "gguf", "gguf", true));
        assertEquals(2, RoutePreflightReport.exitCode("/tmp/model.gguf", null, "gguf", true));
        assertEquals(2, RoutePreflightReport.exitCode(null, "gguf", "gguf", true));
        assertEquals(0, RoutePreflightReport.exitCode(null, null, null, false));
    }

    @Test
    void runCommandJavaNativeFlagSelectsJavaGgufEngine() {
        RunCommand command = new RunCommand();
        command.modelId = "b71c9d";
        command.prompt = "where is jakarta";
        command.javaNativeGguf = true;
        command.maxTokens = 1;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--model", "b71c9d",
                "--prompt", "where is jakarta",
                "--max-tokens", "1",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--engine", "java",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    private static void writeModelIndex(Path home, Path safetensorDir, Path ggufFile) throws Exception {
        Path index = home.resolve("models/index.json");
        Files.createDirectories(index.getParent());
        Files.writeString(index, """
                [
                  {
                    "id": "google/gemma-4-E2B-it",
                    "shortId": "97cbf2",
                    "name": "gemma-4-E2B-it",
                    "format": "safetensors",
                    "path": "%s",
                    "architecture": "gemma"
                  },
                  {
                    "id": "google/gemma-4-E2B-it-Q4_K_M",
                    "shortId": "b71c9d",
                    "name": "gemma-4-E2B-it-Q4_K_M",
                    "format": "gguf",
                    "path": "%s",
                    "architecture": "gemma"
                  }
                ]
                """.formatted(escapeJson(safetensorDir.toString()), escapeJson(ggufFile.toString())));
    }

    private static String escapeJson(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
