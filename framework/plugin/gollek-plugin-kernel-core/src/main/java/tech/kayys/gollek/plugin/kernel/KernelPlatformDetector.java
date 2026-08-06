/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.kernel;

import org.jboss.logging.Logger;

import java.util.*;

/**
 * Kernel platform auto-detection with CPU fallback capability.
 *
 * <p>Automatically detects available kernel platforms and selects the best available option,
 * with support for user override to force CPU usage.</p>
 *
 * <h2>Detection Order</h2>
 * <ol>
 *   <li>Metal (Apple Silicon)</li>
 *   <li>CUDA (NVIDIA GPU)</li>
 *   <li>ROCm (AMD GPU)</li>
 *   <li>DirectML (Windows DirectX)</li>
 *   <li>CPU (Fallback)</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Auto-detect (default)
 * KernelPlatform platform = KernelPlatformDetector.detect();
 *
 * // Force CPU
 * System.setProperty("gollek.kernel.platform", "cpu");
 * KernelPlatform platform = KernelPlatformDetector.detect();
 *
 * // Force specific platform
 * System.setProperty("gollek.kernel.platform", "metal");
 * KernelPlatform platform = KernelPlatformDetector.detect();
 * }</pre>
 *
 * @since 2.0.0
 */
public class KernelPlatformDetector {

    private static final Logger LOG = Logger.getLogger(KernelPlatformDetector.class);

    private static final String PLATFORM_PROPERTY = "gollek.kernel.platform";
    private static final String FORCE_CPU_PROPERTY = "gollek.kernel.force.cpu";
    private static final String CPU_FALLBACK_PROPERTY = "gollek.kernel.cpu.fallback";

    private static KernelPlatformDetector instance;
    private final List<PlatformDetector> detectors = new ArrayList<>();

    /**
     * Platform detector interface.
     */
    public interface PlatformDetector {
        /**
         * Get platform name.
         */
        String getPlatform();

        /**
         * Check if platform is available.
         */
        boolean isAvailable();

        /**
         * Get platform priority (higher = better).
         */
        int getPriority();

        /**
         * Get platform metadata.
         */
        Map<String, String> getMetadata();
    }

    /**
     * Get singleton instance.
     */
    public static synchronized KernelPlatformDetector getInstance() {
        if (instance == null) {
            instance = new KernelPlatformDetector();
        }
        return instance;
    }

    /**
     * Create detector with default detectors.
     */
    private KernelPlatformDetector() {
        // Register detectors in priority order
        registerDetector(new MetalDetector());
        registerDetector(new CudaDetector());
        registerDetector(new RocmDetector());
        registerDetector(new DirectMLDetector());
        registerDetector(new CpuDetector());
    }

    /**
     * Register platform detector.
     */
    public void registerDetector(PlatformDetector detector) {
        detectors.add(detector);
        // Sort by priority (highest first)
        detectors.sort((a, b) -> Integer.compare(b.getPriority(), a.getPriority()));
    }

    /**
     * Detect best available platform.
     *
     * @return detected platform
     */
    public static KernelPlatform detect() {
        return getInstance().detectPlatform();
    }

    /**
     * Detect platform with auto-detection logic.
     */
    private KernelPlatform detectPlatform() {
        // Check if user forced a platform
        String forcedPlatform = System.getProperty(PLATFORM_PROPERTY);
        boolean forceCpu = Boolean.getBoolean(FORCE_CPU_PROPERTY);
        boolean cpuFallback = Boolean.getBoolean(CPU_FALLBACK_PROPERTY);

        if (forceCpu) {
            LOG.info("CPU usage forced via system property");
            return KernelPlatform.CPU;
        }

        if (forcedPlatform != null && !forcedPlatform.trim().isEmpty()) {
            try {
                KernelPlatform platform = KernelPlatform.valueOf(forcedPlatform.trim().toUpperCase());
                LOG.infof("Platform forced to %s via system property", platform);
                return platform;
            } catch (IllegalArgumentException e) {
                LOG.warnf("Invalid platform '%s', will auto-detect", forcedPlatform);
            }
        }

        // Auto-detect best available platform
        for (PlatformDetector detector : detectors) {
            if (detector.isAvailable()) {
                KernelPlatform platform = KernelPlatform.valueOf(detector.getPlatform().toUpperCase());
                LOG.infof("Auto-detected platform: %s (priority: %d)", 
                    platform, detector.getPriority());
                
                Map<String, String> metadata = detector.getMetadata();
                if (!metadata.isEmpty()) {
                    LOG.info("Platform metadata:");
                    metadata.forEach((k, v) -> LOG.infof("  %s: %s", k, v));
                }
                
                return platform;
            }
        }

        // Fallback to CPU
        if (cpuFallback) {
            LOG.info("CPU fallback enabled, using CPU");
        } else {
            LOG.warn("No GPU platform detected, falling back to CPU");
        }
        return KernelPlatform.CPU;
    }

    /**
     * Get all available platforms.
     *
     * @return list of available platforms
     */
    public static List<KernelPlatform> getAvailablePlatforms() {
        List<KernelPlatform> available = new ArrayList<>();
        for (PlatformDetector detector : getInstance().detectors) {
            if (detector.isAvailable()) {
                try {
                    available.add(KernelPlatform.valueOf(detector.getPlatform().toUpperCase()));
                } catch (IllegalArgumentException e) {
                    // Skip unknown platforms
                }
            }
        }
        return available;
    }

    /**
     * Check if platform is available.
     *
     * @param platform platform to check
     * @return true if available
     */
    public static boolean isPlatformAvailable(KernelPlatform platform) {
        for (PlatformDetector detector : getInstance().detectors) {
            if (detector.getPlatform().equalsIgnoreCase(platform.name()) && detector.isAvailable()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get platform metadata.
     *
     * @param platform platform
     * @return metadata map
     */
    public static Map<String, String> getPlatformMetadata(KernelPlatform platform) {
        for (PlatformDetector detector : getInstance().detectors) {
            if (detector.getPlatform().equalsIgnoreCase(platform.name())) {
                return detector.getMetadata();
            }
        }
        return Collections.emptyMap();
    }

    // ───────────────────────────────────────────────────────────────────────
    // Platform Detectors
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Metal detector (Apple Silicon).
     */
    private static class MetalDetector implements PlatformDetector {
        @Override
        public String getPlatform() {
            return "metal";
        }

        @Override
        public boolean isAvailable() {
            try {
                String os = System.getProperty("os.name").toLowerCase();
                String arch = System.getProperty("os.arch");

                if (!os.contains("mac")) {
                    return false;
                }

                // Check for Apple Silicon (M1/M2/M3/M4)
                if (arch.contains("aarch64") || arch.contains("arm64")) {
                    // Metal framework is always present on Apple Silicon Macs —
                    // verify by checking the framework path on disk rather than
                    // requiring a bytedeco wrapper class on the classpath.
                    return java.nio.file.Files.exists(
                            java.nio.file.Path.of("/System/Library/Frameworks/Metal.framework"));
                }

                return false;
            } catch (Exception e) {
                return false;
            }
        }

        @Override
        public int getPriority() {
            return 100; // Highest priority on Apple Silicon
        }

        @Override
        public Map<String, String> getMetadata() {
            Map<String, String> metadata = new HashMap<>();
            try {
                String arch = System.getProperty("os.arch");
                metadata.put("architecture", arch);
                metadata.put("unified_memory", "true");

                // Try to get chip info
                if (arch.contains("aarch64") || arch.contains("arm64")) {
                    metadata.put("chip_type", "Apple Silicon");
                }
            } catch (Exception e) {
                // Ignore
            }
            return metadata;
        }
    }

    /**
     * CUDA detector (NVIDIA GPU).
     */
    private static class CudaDetector implements PlatformDetector {
        @Override
        public String getPlatform() {
            return "cuda";
        }

        @Override
        public boolean isAvailable() {
            try {
                // Check if CUDA is available
                if (!isClassAvailable("org.bytedeco.cuda.cudart.CUDA")) {
                    return false;
                }

                // Check device count
                int[] deviceCount = new int[1];
                Class<?> cudaClass = Class.forName("org.bytedeco.cuda.cudart.CUDA");
                cudaClass.getMethod("cudaGetDeviceCount", int[].class)
                    .invoke(null, new Object[]{deviceCount});

                return deviceCount[0] > 0;
            } catch (Exception e) {
                return false;
            }
        }

        @Override
        public int getPriority() {
            return 90; // High priority
        }

        @Override
        public Map<String, String> getMetadata() {
            Map<String, String> metadata = new HashMap<>();
            try {
                // Get CUDA version
                Class<?> cudaClass = Class.forName("org.bytedeco.cuda.cudart.CUDA");
                int[] version = new int[1];
                cudaClass.getMethod("cudaRuntimeGetVersion", int[].class)
                    .invoke(null, new Object[]{version});

                int cudaVersion = version[0];
                int major = cudaVersion / 1000;
                int minor = (cudaVersion % 1000) / 10;
                metadata.put("cuda_version", major + "." + minor);

                // Get device count
                int[] deviceCount = new int[1];
                cudaClass.getMethod("cudaGetDeviceCount", int[].class)
                    .invoke(null, new Object[]{deviceCount});
                metadata.put("device_count", String.valueOf(deviceCount[0]));
            } catch (Exception e) {
                // Ignore
            }
            return metadata;
        }
    }

    /**
     * ROCm detector (AMD GPU).
     */
    private static class RocmDetector implements PlatformDetector {
        @Override
        public String getPlatform() {
            return "rocm";
        }

        @Override
        public boolean isAvailable() {
            try {
                // Check if ROCm is available
                if (!isClassAvailable("org.bytedeco.rocm.hipRuntime")) {
                    return false;
                }

                // Check for ROCm installation
                String rocmPath = System.getenv("ROCM_PATH");
                if (rocmPath == null) {
                    rocmPath = "/opt/rocm";
                }

                return java.nio.file.Files.exists(java.nio.file.Paths.get(rocmPath));
            } catch (Exception e) {
                return false;
            }
        }

        @Override
        public int getPriority() {
            return 85; // Slightly lower than CUDA
        }

        @Override
        public Map<String, String> getMetadata() {
            Map<String, String> metadata = new HashMap<>();
            try {
                String rocmPath = System.getenv("ROCM_PATH");
                if (rocmPath == null) {
                    rocmPath = "/opt/rocm";
                }
                metadata.put("rocm_path", rocmPath);
            } catch (Exception e) {
                // Ignore
            }
            return metadata;
        }
    }

    /**
     * DirectML detector (Windows DirectX).
     */
    private static class DirectMLDetector implements PlatformDetector {
        @Override
        public String getPlatform() {
            return "directml";
        }

        @Override
        public boolean isAvailable() {
            try {
                String os = System.getProperty("os.name").toLowerCase();
                if (!os.contains("windows")) {
                    return false;
                }

                // Check if DirectML is available
                return isClassAvailable("org.bytedeco.directml.DirectML");
            } catch (Exception e) {
                return false;
            }
        }

        @Override
        public int getPriority() {
            return 80; // Lower priority than CUDA/ROCm
        }

        @Override
        public Map<String, String> getMetadata() {
            Map<String, String> metadata = new HashMap<>();
            metadata.put("os", "Windows");
            metadata.put("directx", "DirectX 12");
            return metadata;
        }
    }

    /**
     * CPU detector (always available).
     */
    private static class CpuDetector implements PlatformDetector {
        @Override
        public String getPlatform() {
            return "cpu";
        }

        @Override
        public boolean isAvailable() {
            return true; // CPU is always available
        }

        @Override
        public int getPriority() {
            return 0; // Lowest priority (fallback)
        }

        @Override
        public Map<String, String> getMetadata() {
            Map<String, String> metadata = new HashMap<>();
            metadata.put("cores", String.valueOf(Runtime.getRuntime().availableProcessors()));
            metadata.put("architecture", System.getProperty("os.arch"));
            return metadata;
        }
    }

    /**
     * Check if class is available.
     */
    private static boolean isClassAvailable(String className) {
        try {
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }
}
