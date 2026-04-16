package tech.kayys.gollek.provider.litert;

import tech.kayys.gollek.spi.model.DeviceType;

import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

final class LiteRTDeviceSupport {

    private LiteRTDeviceSupport() {
    }

    static boolean isAppleSilicon() {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!osName.contains("mac")) {
            return false;
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("aarch64") || arch.contains("arm64");
    }

    static boolean customMetalLibraryExists() {
        String home = System.getProperty("user.home");
        if (home == null) return false;
        return java.nio.file.Files.exists(java.nio.file.Paths.get(home, ".gollek", "libs", "libgollek_metal.dylib"));
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

    static boolean shouldAutoMetal(LiteRTProviderConfig config) {
        return config.autoMetalEnabled() && isAppleSilicon() && metalAllowedByGlobalConfig();
    }

    static boolean effectiveGpuEnabled(LiteRTProviderConfig config) {
        return config.gpuEnabled() || shouldAutoMetal(config);
    }

    static String resolveGpuBackend(LiteRTProviderConfig config) {
        if (!effectiveGpuEnabled(config)) {
            return config.gpuBackend();
        }
        if (shouldAutoMetal(config)) {
            String backend = config.gpuBackend();
            if (backend == null || backend.isBlank() || backend.equalsIgnoreCase("auto")) {
                return "metal";
            }
        }
        return config.gpuBackend();
    }

    static Set<DeviceType> supportedDevices(LiteRTProviderConfig config) {
        LinkedHashSet<DeviceType> devices = new LinkedHashSet<>();
        devices.add(DeviceType.CPU);
        if (effectiveGpuEnabled(config)) {
            devices.add(DeviceType.METAL);
            if (customMetalLibraryExists()) {
                devices.add(DeviceType.NPU);
            }
        }
        return Set.copyOf(devices);
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
