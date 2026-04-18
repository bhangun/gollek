package tech.kayys.gollek.inference.llamacpp;

import tech.kayys.gollek.spi.model.DeviceType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

final class LlamaCppDeviceSupport {

    private LlamaCppDeviceSupport() {
    }

    static boolean isAppleSilicon() {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!osName.contains("mac")) {
            return false;
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("aarch64") || arch.contains("arm64");
    }

    static boolean metalAllowedByGlobalConfig() {
        String metalEnabled = firstNonBlank(
                System.getProperty("gollek.runners.metal.enabled"),
                System.getenv("GOLLEK_METAL_ENABLED"));
        if (metalEnabled != null && metalEnabled.equalsIgnoreCase("false")) {
            return false;
        }
        String metalMode = firstNonBlank(
                System.getProperty("gollek.runners.metal.mode"),
                System.getenv("GOLLEK_METAL_MODE"));
        return metalMode == null || !metalMode.equalsIgnoreCase("disabled");
    }

    static boolean shouldAutoMetal(LlamaCppProviderConfig config) {
        return config.autoMetalEnabled() && isAppleSilicon() && metalAllowedByGlobalConfig() && hasMetalRuntime();
    }

    static boolean effectiveGpuEnabled(LlamaCppProviderConfig config) {
        return config.gpuEnabled() || shouldAutoMetal(config);
    }

    static int resolveGpuLayers(LlamaCppProviderConfig config) {
        if (config.gpuEnabled()) {
            return config.gpuLayers();
        }
        if (shouldAutoMetal(config)) {
            int requested = config.autoMetalLayers();
            return requested == 0 ? -1 : requested;
        }
        return 0;
    }

    static Set<DeviceType> supportedDevices(LlamaCppProviderConfig config) {
        LinkedHashSet<DeviceType> devices = new LinkedHashSet<>();
        devices.add(DeviceType.CPU);
        if (config.gpuEnabled()) {
            devices.add(DeviceType.CUDA);
        }
        if (shouldAutoMetal(config)) {
            devices.add(DeviceType.METAL);
        }
        return Set.copyOf(devices);
    }

    private static boolean hasMetalRuntime() {
        if (!isAppleSilicon()) {
            return false;
        }
        String home = System.getProperty("user.home", "");
        if (home.isBlank()) {
            return false;
        }
        Path base = Path.of(home, ".gollek");
        Path[] candidates = new Path[] {
                base.resolve("libs/libggml-metal.dylib"),
                base.resolve("libs/ggml-metal.dylib"),
                base.resolve("native-libs/libggml-metal.dylib"),
                base.resolve("native-libs/ggml-metal.dylib")
        };
        for (Path candidate : candidates) {
            try {
                if (Files.exists(candidate)) {
                    return true;
                }
            } catch (Exception ignored) {
            }
        }
        return false;
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return null;
    }
}
