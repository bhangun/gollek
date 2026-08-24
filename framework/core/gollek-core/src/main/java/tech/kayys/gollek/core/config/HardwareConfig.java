package tech.kayys.gollek.core.config;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

@ApplicationScoped
public class HardwareConfig {

    @ConfigProperty(name = "hardware.cuda-enabled", defaultValue = "false")
    boolean cudaEnabled;

    @ConfigProperty(name = "hardware.rocm-enabled", defaultValue = "false")
    boolean rocmEnabled;

    @ConfigProperty(name = "hardware.tpu-enabled", defaultValue = "false")
    boolean tpuEnabled;

    @ConfigProperty(name = "hardware.apple-silicon-enabled", defaultValue = "false")
    boolean appleSiliconEnabled;

    @ConfigProperty(name = "hardware.openvino-enabled", defaultValue = "false")
    boolean openVINOEnabled;

    @ConfigProperty(name = "hardware.available-memory", defaultValue = "8589934592")
    long availableMemory;

    public boolean cudaEnabled() {
        return cudaEnabled;
    }

    public boolean rocmEnabled() {
        return rocmEnabled;
    }

    public boolean tpuEnabled() {
        return tpuEnabled;
    }

    public boolean appleSiliconEnabled() {
        return appleSiliconEnabled;
    }

    public boolean openVINOEnabled() {
        return openVINOEnabled;
    }

    public long availableMemory() {
        return availableMemory;
    }
}
