package tech.kayys.gollek.provider.litert;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Lightweight diagnostics for the LiteRT runner stack.
 */
public final class LiteRTRuntimeDiagnostics {

    public static final Set<String> SUPPORTED_FORMATS = Set.of(".litertlm", ".tflite", ".tfl", ".task");

    private LiteRTRuntimeDiagnostics() {
    }

    public static Snapshot snapshot() {
        boolean nativeProviderAvailable = classAvailable("tech.kayys.gollek.provider.litert.LiteRTProvider");
        boolean legacyInterpreterAvailable = classAvailable("org.tensorflow.lite.Interpreter");
        boolean gpuDelegateAvailable = classAvailable("org.tensorflow.lite.gpu.GpuDelegate");
        boolean nnapiAvailable = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("android");
        Optional<Path> nativeLibrary = findNativeLibrary();

        return new Snapshot(
                nativeProviderAvailable,
                LiteRTLmJvmBridge.enabled(),
                LiteRTLmJvmBridge.available(),
                LiteRTLmPythonBridge.enabled(),
                LiteRTLmPythonBridge.available(),
                legacyInterpreterAvailable,
                gpuDelegateAvailable,
                nnapiAvailable,
                nativeLibrary,
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.arch", "unknown"),
                SUPPORTED_FORMATS);
    }

    public static Optional<Path> findNativeLibrary() {
        String override = System.getProperty("LITERT_LIBRARY_PATH");
        if (override == null || override.isBlank()) {
            override = System.getenv("LITERT_LIBRARY_PATH");
        }
        if (override != null && !override.isBlank()) {
            Path overridePath = Path.of(override);
            if (Files.exists(overridePath)) {
                return Optional.of(overridePath);
            }
        }

        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String[] libNames;
        if (os.contains("mac")) {
            libNames = new String[]{"libLiteRt.dylib", "libtensorflowlite_c.dylib"};
        } else if (os.contains("linux")) {
            libNames = new String[]{"libLiteRt.so", "libtensorflowlite_c.so"};
        } else if (os.contains("win")) {
            libNames = new String[]{"LiteRt.dll", "tensorflowlite_c.dll"};
        } else {
            return Optional.empty();
        }

        String home = System.getProperty("user.home");
        Path[] searchDirs = new Path[]{
                Path.of(home, ".gollek", "libs"),
                Path.of("/usr/local/lib"),
                Path.of("/usr/lib"),
                Path.of(home, "lib"),
                Path.of("lib")
        };

        for (Path dir : searchDirs) {
            for (String libName : libNames) {
                Path candidate = dir.resolve(libName);
                if (!Files.exists(candidate)) {
                    continue;
                }
                if (os.contains("mac") && libName.contains("tensorflow")) {
                    Path liteRtCandidate = dir.resolve("libLiteRt.dylib");
                    if (Files.exists(liteRtCandidate)) {
                        return Optional.of(liteRtCandidate);
                    }
                }
                return Optional.of(candidate);
            }
        }

        return Optional.empty();
    }

    private static boolean classAvailable(String className) {
        try {
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException | LinkageError e) {
            return false;
        }
    }

    public record Snapshot(
            boolean providerAvailable,
            boolean officialJvmBridgeEnabled,
            boolean officialJvmBridgeAvailable,
            boolean officialPythonBridgeEnabled,
            boolean officialPythonBridgeAvailable,
            boolean legacyInterpreterAvailable,
            boolean gpuDelegateAvailable,
            boolean nnapiAvailable,
            Optional<Path> nativeLibraryPath,
            String osName,
            String osArch,
            Set<String> supportedFormats) {

        public boolean hasExecutionRuntime() {
            return providerAvailable
                    && (officialJvmBridgeAvailable
                    || officialPythonBridgeAvailable
                    || nativeLibraryPath.isPresent()
                    || legacyInterpreterAvailable);
        }

        public Map<String, Object> asMap() {
            Map<String, Object> values = new LinkedHashMap<>();
            values.put("provider_available", providerAvailable);
            values.put("official_litertlm_jvm_bridge_enabled", officialJvmBridgeEnabled);
            values.put("official_litertlm_jvm_bridge_available", officialJvmBridgeAvailable);
            values.put("official_litertlm_python_bridge_enabled", officialPythonBridgeEnabled);
            values.put("official_litertlm_python_bridge_available", officialPythonBridgeAvailable);
            values.put("legacy_tflite_interpreter_available", legacyInterpreterAvailable);
            values.put("gpu_delegate_available", gpuDelegateAvailable);
            values.put("nnapi_available", nnapiAvailable);
            values.put("native_library_path", nativeLibraryPath.map(Path::toString).orElse("unavailable"));
            values.put("has_execution_runtime", hasExecutionRuntime());
            values.put("os", osName);
            values.put("arch", osArch);
            values.put("supported_formats", supportedFormats);
            return values;
        }
    }
}
