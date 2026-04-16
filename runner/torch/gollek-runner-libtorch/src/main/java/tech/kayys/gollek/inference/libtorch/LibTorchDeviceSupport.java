package tech.kayys.gollek.inference.libtorch;

import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.spi.model.DeviceType;

import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

final class LibTorchDeviceSupport {

    private LibTorchDeviceSupport() {
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

    static boolean shouldAutoMps(LibTorchProviderConfig config) {
        return config.gpu().autoMpsEnabled() && isAppleSilicon() && metalAllowedByGlobalConfig();
    }

    static Device resolveDevice(LibTorchProviderConfig config) {
        if (config.gpu().enabled()) {
            return Device.cuda(config.gpu().deviceIndex());
        }
        if (shouldAutoMps(config)) {
            return Device.MPS;
        }
        return Device.CPU;
    }

    static Set<DeviceType> supportedDevices(LibTorchProviderConfig config) {
        LinkedHashSet<DeviceType> devices = new LinkedHashSet<>();
        devices.add(DeviceType.CPU);
        if (config.gpu().enabled()) {
            devices.add(DeviceType.CUDA);
        }
        if (shouldAutoMps(config)) {
            devices.add(DeviceType.METAL);
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
