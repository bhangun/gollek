package tech.kayys.golek.model.core;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

/**
 * Hardware capability detector
 */
@ApplicationScoped
public class HardwareDetector {

    @ConfigProperty(name = "hardware.cuda.enabled", defaultValue = "false")
    boolean cudaEnabled;

    @ConfigProperty(name = "hardware.memory.available", defaultValue = "8589934592") // 8GB
    long availableMemory;

    public HardwareCapabilities detect() {
        return HardwareCapabilities.builder()
                .hasCUDA(cudaEnabled && isCUDAAvailable())
                .availableMemory(availableMemory)
                .cpuCores(Runtime.getRuntime().availableProcessors())
                .build();
    }

    private boolean isCUDAAvailable() {
        if (!cudaEnabled) {
            return false;
        }

        // 1. Check for NVIDIA Management Library (NVML) which is part of the driver
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("win") ? "nvml.dll" : "libnvidia-ml.so";

        // Try to find the library in common locations or system path
        // This is a heuristic but fairly reliable for driver presence
        boolean hasLib = false;
        if (os.contains("win")) {
            hasLib = java.nio.file.Files
                    .exists(java.nio.file.Paths.get(System.getenv("SystemRoot"), "System32", libName));
        } else {
            hasLib = java.nio.file.Files.exists(java.nio.file.Paths.get("/usr/lib/x86_64-linux-gnu", libName)) ||
                    java.nio.file.Files.exists(java.nio.file.Paths.get("/usr/lib64", libName));
        }

        if (hasLib) {
            return true;
        }

        // 2. Fallback: Try executing nvidia-smi
        try {
            Process process = new ProcessBuilder("nvidia-smi", "-L").start();
            return process.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

}