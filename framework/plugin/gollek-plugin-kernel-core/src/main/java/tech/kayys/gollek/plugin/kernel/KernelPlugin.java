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

import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletionStage;

/**
 * Enhanced SPI for platform-specific kernel plugins with comprehensive lifecycle
 * management, error handling, and observability support.
 *
 * <p>Kernel plugins provide platform-specific GPU kernel implementations for:
 * <ul>
 *   <li><b>CUDA</b> - NVIDIA GPUs (Compute Capability 6.0+)</li>
 *   <li><b>ROCm</b> - AMD GPUs (CDNA/RDNA architectures)</li>
 *   <li><b>Metal</b> - Apple Silicon (M1/M2/M3 series)</li>
 *   <li><b>DirectML</b> - Windows DirectX 12 (Any GPU)</li>
 *   <li><b>CPU</b> - Fallback for systems without GPU acceleration</li>
 * </ul>
 *
 * <h2>Plugin Lifecycle</h2>
 * <pre>
 * ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
 * │ LOADED  │ →  │ VALIDATING│ →  │ VALIDATED │ →  │INITIALIZING│ →  │ ACTIVE  │
 * └─────────┘    └──────────┘    └───────────┘    └──────────┘    └─────────┘
 *                                                               ↓
 *                                                          ┌─────────┐
 *                                                          │ ERROR   │
 *                                                          └─────────┘
 *                                                               ↓
 *                                                          ┌─────────┐
 *                                                          │STOPPED  │
 *                                                          └─────────┘
 * </pre>
 *
 * <h2>Example Implementation</h2>
 * <pre>{@code
 * public class CudaKernelPlugin implements KernelPlugin {
 *
 *     {@code @Override}
 *     public String id() { return "cuda-kernel"; }
 *
 *     {@code @Override}
 *     public KernelConfig getConfig() {
 *         return KernelConfig.builder()
 *             .deviceId(0)
 *             .memoryFraction(0.9f)
 *             .allowGrowth(true)
 *             .build();
 *     }
 *
 *     {@code @Override}
 *     public void initialize(KernelContext context) throws KernelException {
 *         // Initialize CUDA runtime
 *         cudaInit(context.getConfig());
 *     }
 *
 *     {@code @Override}
 *     public <T> KernelResult<T> execute(KernelOperation operation,
 *                                        KernelContext context) {
 *         return switch (operation.getName()) {
 *             case "gemm" -> executeGemm(operation, context);
 *             case "attention" -> executeAttention(operation, context);
 *             default -> throw new UnknownOperationException(operation.getName());
 *         };
 *     }
 * }
 * }</pre>
 *
 * <h2>Platform Detection</h2>
 * <pre>{@code
 * // Auto-detect platform
 * String platform = KernelPlugin.autoDetectPlatform();
 *
 * // Load appropriate kernel plugin
 * KernelPlugin kernel = KernelPluginManager.getInstance()
 *     .getPluginForPlatform(platform)
 *     .orElseThrow(() -> new KernelNotFoundException(platform));
 * }</pre>
 *
 * @since 2.1.0
 * @version 2.0.0 (Enhanced with lifecycle, error handling, and observability)
 */
public interface KernelPlugin {

    // ========================================================================
    // Plugin Identity
    // ========================================================================

    /**
     * Unique kernel plugin identifier.
     *
     * @return kernel ID (e.g., "cuda-kernel", "rocm-kernel", "metal-kernel")
     */
    String id();

    /**
     * Human-readable name.
     *
     * @return kernel name (e.g., "CUDA Kernel", "Metal Kernel")
     */
    String name();

    /**
     * Kernel version following semantic versioning.
     *
     * @return version string (e.g., "1.0.0", "2.1.3")
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Kernel description.
     *
     * @return detailed description of kernel capabilities
     */
    String description();

    /**
     * Get the target platform for this kernel.
     *
     * @return platform ID (e.g., "cuda", "rocm", "metal", "directml", "cpu")
     */
    String platform();

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /**
     * Validate kernel prerequisites before initialization.
     *
     * <p>Called during plugin validation phase to ensure all requirements
     * are met (drivers, libraries, hardware).</p>
     *
     * @return validation result with status and any error messages
     */
    default KernelValidationResult validate() {
        try {
            boolean available = isAvailable();
            if (available) {
                return KernelValidationResult.valid();
            } else {
                return KernelValidationResult.invalid(
                    "Kernel not available on this platform");
            }
        } catch (Exception e) {
            return KernelValidationResult.invalid(
                "Validation failed: " + e.getMessage());
        }
    }

    /**
     * Initialize the kernel with context and configuration.
     *
     * <p>Called during plugin initialization phase. Should prepare all
     * resources needed for kernel execution.</p>
     *
     * @param context Kernel context with configuration and services
     * @throws KernelException if initialization fails
     */
    default void initialize(KernelContext context) throws KernelException {
        // Default: no-op
    }

    /**
     * Check if this kernel is available on current platform.
     *
     * <p>Performs quick availability check without detailed validation.</p>
     *
     * @return true if kernel can potentially be used
     */
    boolean isAvailable();

    /**
     * Check if kernel is healthy and operational.
     *
     * <p>Called periodically during runtime to monitor kernel health.</p>
     *
     * @return true if kernel is healthy and ready for operations
     */
    default boolean isHealthy() {
        return true;
    }

    /**
     * Get detailed health information.
     *
     * @return health status with details
     */
    default KernelHealth health() {
        boolean healthy = isHealthy();
        return healthy ?
            KernelHealth.healthy() :
            KernelHealth.unhealthy("Kernel reported unhealthy");
    }

    /**
     * Shutdown and cleanup resources.
     *
     * <p>Called when plugin is being unloaded. Should release all resources
     * and perform cleanup.</p>
     */
    default void shutdown() {
        // Default: no-op
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    /**
     * Execute a kernel operation synchronously.
     *
     * @param operation Operation to execute
     * @param context Execution context with parameters and metadata
     * @param <T> Expected result type
     * @return execution result
     * @throws KernelException if execution fails
     */
    <T> KernelResult<T> execute(KernelOperation operation,
                                KernelContext context) throws KernelException;

    /**
     * Execute a kernel operation asynchronously.
     *
     * @param operation Operation to execute
     * @param context Execution context with parameters and metadata
     * @param <T> Expected result type
     * @return completion stage with result
     */
    default <T> CompletionStage<KernelResult<T>> executeAsync(
            KernelOperation operation,
            KernelContext context) {
        // Default: wrap synchronous execution
        try {
            KernelResult<T> result = execute(operation, context);
            return java.util.concurrent.CompletableFuture.completedFuture(result);
        } catch (KernelException e) {
            java.util.concurrent.CompletableFuture<KernelResult<T>> future =
                new java.util.concurrent.CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }

    /**
     * Get supported operations for this kernel.
     *
     * @return set of supported operation names
     */
    default Set<String> supportedOperations() {
        return Collections.emptySet();
    }

    // ========================================================================
    // Capabilities and Metadata
    // ========================================================================

    /**
     * Get supported device architectures for this platform.
     *
     * @return set of supported architectures (e.g., "ampere", "hopper", "m1", "m2")
     */
    Set<String> supportedArchitectures();

    /**
     * Get supported compute capabilities/versions.
     *
     * @return set of supported versions (e.g., "8.0", "9.0", "metal_3_0")
     */
    Set<String> supportedVersions();

    /**
     * Get kernel-specific metadata.
     *
     * @return metadata map with capabilities, versions, and features
     */
    default Map<String, Object> metadata() {
        return Map.of(
            "platform", platform(),
            "version", version(),
            "architectures", supportedArchitectures(),
            "compute_versions", supportedVersions(),
            "operations", supportedOperations()
        );
    }

    /**
     * Get kernel configuration.
     *
     * @return kernel configuration
     */
    default KernelConfig getConfig() {
        return KernelConfig.defaultConfig();
    }

    /**
     * Get plugin dependencies.
     *
     * @return map of plugin ID to version requirement
     */
    default Map<String, String> dependencies() {
        return Collections.emptyMap();
    }

    // ========================================================================
    // Platform Detection
    // ========================================================================

    /**
     * Auto-detect the current platform.
     *
     * <p>Detection order:
     * <ol>
     *   <li>Apple Silicon (Metal) - macOS on ARM</li>
     *   <li>Windows (DirectML) - Windows OS</li>
     *   <li>CUDA - NVIDIA GPU with CUDA libraries</li>
     *   <li>ROCm - AMD GPU with ROCm libraries</li>
     *   <li>CPU - Fallback for systems without GPU</li>
     * </ol>
     *
     * @return platform ID ("cuda", "rocm", "metal", "directml", or "cpu")
     */
    static String autoDetectPlatform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();

        // Check for Apple Silicon (Metal)
        if (os.contains("mac")) {
            if (arch.contains("aarch64") || arch.contains("arm64")) {
                return "metal";
            }
            // Intel Mac - still use Metal if available
            return "metal";
        }

        // Check for Windows (DirectML)
        if (os.contains("windows")) {
            return "directml";
        }

        // Check for Linux
        if (os.contains("linux")) {
            // Check for NVIDIA CUDA
            if (isClassAvailable("org.bytedeco.cuda.cudart.CUDA")) {
                return "cuda";
            }

            // Check for AMD ROCm
            if (isClassAvailable("org.bytedeco.rocm.hipRuntime") ||
                isClassAvailable("com.amd.acl.ACL")) {
                return "rocm";
            }
        }

        // Default to CPU
        return "cpu";
    }

    /**
     * Check if a class is available in the classpath.
     *
     * @param className fully qualified class name
     * @return true if class can be loaded
     */
    static boolean isClassAvailable(String className) {
        try {
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }

    /**
     * Detect device architecture.
     *
     * @return detected architecture or "unknown"
     */
    static String detectArchitecture() {
        String arch = System.getProperty("os.arch").toLowerCase();

        if (arch.contains("aarch64") || arch.contains("arm64")) {
            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                // Apple Silicon - try to detect specific model
                String cpuBrand = System.getProperty("hw.cpufamily.brand", "unknown");
                if (cpuBrand.contains("M3")) return "m3";
                if (cpuBrand.contains("M2")) return "m2";
                if (cpuBrand.contains("M1")) return "m1";
                return "apple_silicon";
            }
            return "arm64";
        }

        if (arch.contains("amd64") || arch.contains("x86_64")) {
            return "x86_64";
        }

        return arch;
    }
}
