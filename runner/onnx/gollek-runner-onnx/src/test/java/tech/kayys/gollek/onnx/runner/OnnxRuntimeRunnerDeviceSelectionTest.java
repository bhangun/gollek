package tech.kayys.gollek.onnx.runner;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

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
        List<tech.kayys.gollek.spi.model.DeviceType> devices =
                (List<tech.kayys.gollek.spi.model.DeviceType>) method.invoke(runner);

        assertTrue(devices.contains(tech.kayys.gollek.spi.model.DeviceType.METAL));
    }

    private boolean invokeMetalAllowed(OnnxRuntimeRunner runner) throws Exception {
        Method method = OnnxRuntimeRunner.class.getDeclaredMethod("metalAllowedByGlobalConfig");
        method.setAccessible(true);
        return (boolean) method.invoke(runner);
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
}
