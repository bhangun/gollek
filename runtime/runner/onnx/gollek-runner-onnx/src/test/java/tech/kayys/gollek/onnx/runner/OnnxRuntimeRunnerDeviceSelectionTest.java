package tech.kayys.gollek.onnx.runner;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxRuntimeRunnerDeviceSelectionTest {

    @TempDir
    Path tempDir;

    private String originalOsName;
    private String originalArch;
    private String originalMetalEnabled;
    private String originalMetalMode;
    private String originalUserHome;

    @BeforeEach
    void captureProperties() {
        originalOsName = System.getProperty("os.name");
        originalArch = System.getProperty("os.arch");
        originalMetalEnabled = System.getProperty("gollek.runners.metal.enabled");
        originalMetalMode = System.getProperty("gollek.runners.metal.mode");
        originalUserHome = System.getProperty("user.home");
    }

    @AfterEach
    void restoreProperties() {
        restoreProperty("os.name", originalOsName);
        restoreProperty("os.arch", originalArch);
        restoreProperty("gollek.runners.metal.enabled", originalMetalEnabled);
        restoreProperty("gollek.runners.metal.mode", originalMetalMode);
        restoreProperty("user.home", originalUserHome);
    }

    @Test
    void coreMlSkippedWhenGlobalMetalDisabled() throws Exception {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");
        System.setProperty("gollek.runners.metal.enabled", "false");

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        assertFalse(invokeMetalAllowed(runner));
    }

    @Test
    void coreMlAllowedWhenGlobalMetalEnabled() throws Exception {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");
        System.setProperty("gollek.runners.metal.enabled", "true");
        System.setProperty("gollek.runners.metal.mode", "auto");

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        assertTrue(invokeMetalAllowed(runner));
    }

    @Test
    void coreMlSkippedWhenModeDisabled() throws Exception {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");
        System.setProperty("gollek.runners.metal.mode", "disabled");

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        assertFalse(invokeMetalAllowed(runner));
    }

    @Test
    void resolveLibraryPathFallsBackToGollekLibs() throws Exception {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("user.home", tempDir.toString());

        Path fallback = tempDir.resolve(".gollek/libs/libonnxruntime.dylib");
        Files.createDirectories(fallback.getParent());
        Files.writeString(fallback, "onnx");

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        Method method = OnnxRuntimeRunner.class.getDeclaredMethod("resolveLibraryPath", String.class);
        method.setAccessible(true);

        Path resolved = (Path) method.invoke(runner, "/nope/libonnxruntime.dylib");
        assertEquals(fallback, resolved);
    }

    @Test
    void supportedDevicesIncludeMetalWhenCoreMlAvailableOnAppleSilicon() throws Exception {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "coreMlAvailable", true);

        Method method = OnnxRuntimeRunner.class.getDeclaredMethod("resolveSupportedDevices");
        method.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<tech.kayys.alkhawarizm.core.tensor.DeviceType> devices = (List<tech.kayys.alkhawarizm.core.tensor.DeviceType>) method
                .invoke(runner);

        assertTrue(devices.contains(tech.kayys.alkhawarizm.core.tensor.DeviceType.METAL));
    }

    @Test
    void metalAliasesResolveToMetalDeviceType() throws Exception {
        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "executionProvider", "metal");

        assertEquals(tech.kayys.alkhawarizm.core.tensor.DeviceType.METAL, runner.deviceType());

        setField(runner, "executionProvider", "mps");
        assertEquals(tech.kayys.alkhawarizm.core.tensor.DeviceType.METAL, runner.deviceType());
    }

    @Test
    void textRuntimePlanPrecomputesKvAndPositionNameSets() throws Exception {
        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "usePastKvCache", true);
        setField(runner, "kvLayers", 2);
        setField(runner, "inputNames", new String[] {
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value"
        });
        setField(runner, "outputNames", new String[] {
                "logits",
                "present.0.key",
                "present.0.value",
                "present.1.key",
                "present.1.value"
        });

        Object plan = invokeNoArg(runner, "buildTextRuntimePlan");

        assertTrue((boolean) invokeNoArg(plan, "hasKvInputs"));
        assertTrue((boolean) invokeNoArg(plan, "hasPositionIds"));
        assertEquals(2, invokeNoArg(plan, "kvLayerCount"));
        assertArrayEquals(new String[] {
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value"
        }, (String[]) invokeNoArg(plan, "runInputNames"));
        assertArrayEquals(new String[] {
                "logits",
                "present.0.key",
                "present.0.value",
                "present.1.key",
                "present.1.value"
        }, (String[]) invokeNoArg(plan, "runOutputNames"));
    }

    @Test
    void textRuntimePlanDisablesKvWhenPresentOutputsAreIncomplete() throws Exception {
        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "usePastKvCache", true);
        setField(runner, "kvLayers", 1);
        setField(runner, "inputNames", new String[] {
                "input_ids",
                "attention_mask",
                "past_key_values.0.key",
                "past_key_values.0.value"
        });
        setField(runner, "outputNames", new String[] { "logits", "present.0.key" });

        Object plan = invokeNoArg(runner, "buildTextRuntimePlan");

        assertFalse((boolean) invokeNoArg(plan, "hasKvInputs"));
        assertFalse((boolean) invokeNoArg(plan, "hasPositionIds"));
        assertEquals(0, invokeNoArg(plan, "kvLayerCount"));
        assertArrayEquals(new String[] { "input_ids", "attention_mask" },
                (String[]) invokeNoArg(plan, "runInputNames"));
        assertArrayEquals(new String[] { "logits" },
                (String[]) invokeNoArg(plan, "runOutputNames"));
    }

    @Test
    void eosCheckUsesPrecomputedStopTokenSet() throws Exception {
        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        setField(runner, "eosTokenIds", OnnxStopTokens.of(42));
        setField(runner, "tokenizer", new ThrowingStopTokenizer());

        assertTrue(runner.isEos(42));
        assertFalse(runner.isEos(43));
    }

    @Test
    void genAiConfigStopTokensKeepDefaultEosToken() throws Exception {
        Files.writeString(tempDir.resolve("genai_config.json"), """
                {
                  "model": {
                    "eos_token_id": 42,
                    "decoder": {
                      "num_hidden_layers": 1
                    }
                  }
                }
                """);

        OnnxRuntimeRunner runner = new OnnxRuntimeRunner();
        invoke(runner, "loadGenAiConfig", Path.class, tempDir);

        assertTrue(runner.isEos(2));
        assertTrue(runner.isEos(42));
        assertFalse(runner.isEos(43));
    }

    private boolean invokeMetalAllowed(OnnxRuntimeRunner runner) throws Exception {
        Method method = OnnxRuntimeRunner.class.getDeclaredMethod("metalAllowedByGlobalConfig");
        method.setAccessible(true);
        return (boolean) method.invoke(runner);
    }

    private static Object invokeNoArg(Object target, String methodName) throws Exception {
        Method method = target.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        return method.invoke(target);
    }

    private static void invoke(Object target, String methodName, Class<?> parameterType, Object argument)
            throws Exception {
        Method method = target.getClass().getDeclaredMethod(methodName, parameterType);
        method.setAccessible(true);
        method.invoke(target, argument);
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        java.lang.reflect.Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static final class ThrowingStopTokenizer implements Tokenizer {

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            return "";
        }

        @Override
        public int vocabSize() {
            return 0;
        }

        @Override
        public int bosTokenId() {
            return -1;
        }

        @Override
        public int eosTokenId() {
            return -1;
        }

        @Override
        public int padTokenId() {
            return -1;
        }

        @Override
        public int[] allStopTokenIds() {
            throw new AssertionError("isEos should use the precomputed stop-token set");
        }
    }
}
