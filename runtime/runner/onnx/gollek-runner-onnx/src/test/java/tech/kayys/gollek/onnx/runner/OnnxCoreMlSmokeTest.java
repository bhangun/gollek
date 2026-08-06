package tech.kayys.gollek.onnx.runner;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class OnnxCoreMlSmokeTest {

    @Test
    void coreMlAvailableOnAppleSiliconWhenSymbolPresent() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        assumeTrue(osName.contains("mac") && (arch.contains("arm64") || arch.contains("aarch64")),
                "Apple Silicon required for CoreML EP");

        String libPath = resolveLibraryPath();
        assumeTrue(libPath != null && !libPath.isBlank(), "Set gollek.onnx.library-path or GOLLEK_ONNX_LIBRARY_PATH");

        boolean initialized = tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding.initialize(Path.of(libPath));
        assumeTrue(initialized, "ONNX Runtime native library not available");

        boolean hasCoreMl = tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding.getInstance().supportsCoreMl();
        if ("true".equalsIgnoreCase(System.getProperty("gollek.onnx.coreml.required", "false"))) {
            assertTrue(hasCoreMl, "CoreML EP symbol missing");
        } else {
            assertFalse(!hasCoreMl && "true".equalsIgnoreCase(System.getProperty("gollek.onnx.coreml.optional", "true")),
                    "CoreML EP symbol missing");
        }
    }

    private String resolveLibraryPath() {
        String configured = System.getProperty("gollek.onnx.library-path",
                System.getenv("GOLLEK_ONNX_LIBRARY_PATH"));
        if (configured != null && !configured.isBlank()) {
            Path configuredPath = Path.of(configured);
            if (java.nio.file.Files.exists(configuredPath) && java.nio.file.Files.isReadable(configuredPath)) {
                return configured;
            }
        }
        Path fallback = Path.of(System.getProperty("user.home"), ".gollek", "libs", "libonnxruntime.dylib");
        if (java.nio.file.Files.exists(fallback)) {
            return fallback.toString();
        }
        return configured;
    }
}
