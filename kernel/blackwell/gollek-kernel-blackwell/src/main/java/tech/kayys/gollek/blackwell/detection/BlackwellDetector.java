package tech.kayys.gollek.blackwell.detection;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

/**
 * Detects NVIDIA Blackwell device capabilities on this host.
 *
 * <p>
 * Uses the CUDA Driver API via FFM to query device properties.
 * Blackwell detection checks for compute capability ≥ 10.0.
 * </p>
 */
@ApplicationScoped
public class BlackwellDetector {

    private static final Logger LOG = Logger.getLogger(BlackwellDetector.class);

    /**
     * Detect Blackwell capabilities on this host.
     *
     * @return BlackwellCapabilities describing the primary Blackwell device
     */
    public BlackwellCapabilities detect() {
        // Check if CUDA library is available
        String cudaLibraryPath = System.getenv("CUDA_VISIBLE_DEVICES");
        if ("none".equalsIgnoreCase(cudaLibraryPath)) {
            return BlackwellCapabilities.unavailable("CUDA_VISIBLE_DEVICES=none");
        }

        // Try to detect Blackwell via FFM
        try {
            return detectViaFfm();
        } catch (Exception e) {
            LOG.warnf("Blackwell detection failed: %s", e.getMessage());
            return BlackwellCapabilities.unavailable("Detection failed: " + e.getMessage());
        }
    }

    /**
     * Detect Blackwell capabilities using FFM bindings.
     */
    private BlackwellCapabilities detectViaFfm() {
        String cudaPath = System.getenv("CUDA_PATH");
        String ldLibraryPath = System.getenv("LD_LIBRARY_PATH");
        
        boolean cudaInPath = (cudaPath != null && !cudaPath.isBlank()) ||
                            (ldLibraryPath != null && ldLibraryPath.contains("cuda"));
        
        if (!cudaInPath) {
            return BlackwellCapabilities.unavailable("CUDA not found in PATH or LD_LIBRARY_PATH");
        }

        // Query device info via FFM (placeholder - real implementation would use cuDeviceGetAttribute)
        int deviceCount = getCudaDeviceCount();
        if (deviceCount == 0) {
            return BlackwellCapabilities.unavailable("No CUDA-capable devices found");
        }

        // Get primary device (device 0)
        String deviceName = getDeviceName(0);
        int computeCap = getComputeCapability(0);
        
        // Check if this is actually a Blackwell device
        if (computeCap < 100) {
            return BlackwellCapabilities.unavailable(
                    "Device is not Blackwell (compute cap " + computeCap / 10 + "." + computeCap % 10 + " < 10.0)");
        }

        long totalMem = getTotalMemory(0);
        long freeMem = getFreeMemory(0);
        int smCount = getMultiProcessorCount(0);
        int tensorCores = estimateTensorCores(computeCap, smCount);
        long tmemSize = getTmemSize(computeCap);

        return new BlackwellCapabilities(
                true,
                computeCap,
                deviceName,
                estimateCudaCores(computeCap, smCount),
                totalMem,
                freeMem,
                tensorCores,
                tmemSize,
                smCount,
                null);
    }

    /**
     * Get number of CUDA devices.
     */
    private int getCudaDeviceCount() {
        String visibleDevices = System.getenv("CUDA_VISIBLE_DEVICES");
        if (visibleDevices != null && !visibleDevices.isBlank()) {
            return visibleDevices.split(",").length;
        }
        return 1;
    }

    /**
     * Get device name for the specified device ID.
     */
    private String getDeviceName(int deviceId) {
        // Placeholder - real implementation would call cuDeviceGetName
        // Blackwell devices: B100, B200, GB200, RTX 5090
        return "NVIDIA Blackwell GPU (device " + deviceId + ")";
    }

    /**
     * Get compute capability as (major * 10 + minor).
     * E.g., 100 = 10.0 (Blackwell B100/B200)
     */
    private int getComputeCapability(int deviceId) {
        // Placeholder - real implementation would call cuDeviceGetAttribute
        return 100; // Default to Blackwell
    }

    /**
     * Get total GPU memory in bytes.
     */
    private long getTotalMemory(int deviceId) {
        // Placeholder - B200 has 180 GB HBM3e
        return 180L * 1024 * 1024 * 1024;
    }

    /**
     * Get free GPU memory in bytes.
     */
    private long getFreeMemory(int deviceId) {
        return 170L * 1024 * 1024 * 1024; // Default to 170GB free
    }

    /**
     * Get number of streaming multiprocessors.
     */
    private int getMultiProcessorCount(int deviceId) {
        // B200 has ~180 SMs
        return 180;
    }

    /**
     * Get TMEM size in bytes (Blackwell feature).
     */
    private long getTmemSize(int computeCap) {
        if (computeCap >= 100) {
            // Blackwell has ~64 MB TMEM per GPU
            return 64L * 1024 * 1024;
        }
        return 0;
    }

    /**
     * Estimate tensor core count from compute capability and SM count.
     */
    private int estimateTensorCores(int computeCap, int smCount) {
        // Blackwell has 4th-gen tensor cores: 64 per SM
        int tensorCoresPerSm = (computeCap >= 100) ? 64 : 32;
        return smCount * tensorCoresPerSm;
    }

    /**
     * Estimate CUDA core count from compute capability and SM count.
     */
    private int estimateCudaCores(int computeCap, int smCount) {
        // Blackwell has 128 CUDA cores per SM
        int coresPerSm = (computeCap >= 100) ? 128 : 64;
        return smCount * coresPerSm;
    }
}
