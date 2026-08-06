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

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Enhanced manager for platform-specific kernel plugins with comprehensive
 * lifecycle management, observability, and error handling.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>Auto-detect platform (CUDA, ROCm, Metal, DirectML)</li>
 *   <li>Load only kernel for current platform</li>
 *   <li>Fallback to CPU if no GPU available</li>
 *   <li>Health monitoring and metrics</li>
 *   <li>Hot-reload support</li>
 *   <li>Dependency resolution</li>
 * </ul>
 *
 * <h2>Lifecycle:</h2>
 * <pre>
 * LOADED → VALIDATING → VALIDATED → INITIALIZING → ACTIVE → STOPPED
 *              ↓              ↓            ↓
 *           ERROR          ERROR       ERROR
 * </pre>
 *
 * @since 2.1.0
 * @version 2.0.0 (Enhanced with lifecycle, observability, and error handling)
 */
@ApplicationScoped
public class KernelPluginManager {

    private static final Logger LOG = Logger.getLogger(KernelPluginManager.class);

    @Inject
    Instance<KernelPlugin> kernelPluginInstances;

    private final Map<String, KernelPlugin> kernels = new ConcurrentHashMap<>();
    private final Map<String, KernelPluginState> kernelStates = new ConcurrentHashMap<>();
    private final List<KernelListener> listeners = new CopyOnWriteArrayList<>();

    private KernelPlugin activeKernel = null;
    private String activePlatform = null;
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final KernelMetrics metrics = new KernelMetrics();

    /**
     * Initialize the kernel plugin manager by detecting platform and loading
     * appropriate kernel.
     */
    @jakarta.annotation.PostConstruct
    public void initialize() {
        if (initialized.compareAndSet(false, true)) {
            doInitialize();
        } else {
            LOG.warn("Kernel plugin manager already initialized");
        }
    }

    private void doInitialize() {
        LOG.info("Initializing kernel plugin manager...");
        metrics.recordEvent("initialization_started");

        try {
            // Auto-detect platform
            String platform = KernelPlugin.autoDetectPlatform();
            LOG.infof("Auto-detected platform: %s", platform);
            activePlatform = platform;

            // Discover and register kernels
            if (kernelPluginInstances != null) {
                for (KernelPlugin kernel : kernelPluginInstances) {
                    try {
                        register(kernel);
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to register kernel plugin: %s", kernel.id());
                        metrics.recordError("kernel_registration_failed", e);
                    }
                }
            }

            // Validate all kernels
            validateAllKernels();

            // Select and initialize kernel for current platform
            selectKernelForPlatform(platform);

            metrics.recordEvent("initialization_completed");
            LOG.infof("Kernel plugin manager initialized. Active kernel: %s",
                    activeKernel != null ? activeKernel.id() : "cpu");

        } catch (KernelInitializationException e) {
            LOG.errorf(e, "Failed to initialize kernel plugin manager");
            metrics.recordError("initialization_failed", e);
            initialized.set(false);
            throw e;
        } catch (Exception e) {
            LOG.errorf(e, "Failed to initialize kernel plugin manager");
            metrics.recordError("initialization_failed", e);
            initialized.set(false);
            throw new KernelInitializationException("Failed to initialize kernel manager", e);
        }
    }

    /**
     * Register a kernel plugin.
     *
     * @param kernel kernel plugin to register
     * @throws KernelException if registration fails
     */
    public void register(KernelPlugin kernel) throws KernelException {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null");
        }

        LOG.infof("Registering kernel plugin: %s (platform: %s)",
                kernel.id(), kernel.platform());

        kernels.put(kernel.platform(), kernel);
        kernelStates.put(kernel.platform(), KernelPluginState.LOADED);

        notifyKernelRegistered(kernel);
        metrics.recordKernelRegistered(kernel.platform());
    }

    /**
     * Validate all registered kernels.
     */
    public void validateAllKernels() {
        LOG.info("Validating all kernels...");

        for (Map.Entry<String, KernelPlugin> entry : kernels.entrySet()) {
            String platform = entry.getKey();
            KernelPlugin kernel = entry.getValue();

            try {
                KernelValidationResult result = kernel.validate();
                if (result.isValid()) {
                    kernelStates.put(platform, KernelPluginState.VALIDATED);
                    LOG.infof("Kernel %s validated successfully", platform);
                } else {
                    kernelStates.put(platform, KernelPluginState.ERROR);
                    LOG.warnf("Kernel %s validation failed: %s",
                            platform, result.getFirstError().orElse("Unknown error"));
                    metrics.recordValidationError(platform, result.getErrors());
                }
            } catch (Exception e) {
                kernelStates.put(platform, KernelPluginState.ERROR);
                LOG.errorf(e, "Kernel %s validation threw exception", platform);
                metrics.recordValidationError(platform, List.of(e.getMessage()));
            }
        }
    }

    /**
     * Select kernel for specified platform.
     *
     * @param platform platform to select
     * @throws KernelException if selection fails
     */
    public void selectKernelForPlatform(String platform) throws KernelException {
        LOG.infof("Selecting kernel for platform: %s", platform);

        KernelPlugin kernel = kernels.get(platform);
        if (kernel != null && kernel.isAvailable()) {
            try {
                // Initialize kernel
                kernelStates.put(platform, KernelPluginState.INITIALIZING);
                KernelContext context = KernelContext.builder()
                        .config(kernel.getConfig())
                        .metadata("platform", platform)
                        .build();
                kernel.initialize(context);

                activeKernel = kernel;
                activePlatform = platform;
                kernelStates.put(platform, KernelPluginState.ACTIVE);

                LOG.infof("Selected and initialized kernel for platform %s: %s",
                        platform, kernel.id());
                notifyKernelActivated(kernel);
                metrics.recordKernelActivated(platform);

            } catch (Exception e) {
                kernelStates.put(platform, KernelPluginState.ERROR);
                LOG.errorf(e, "Failed to initialize kernel for platform %s", platform);
                metrics.recordError("kernel_initialization_failed", e);

                // Try fallback to CPU
                if (!"cpu".equals(platform)) {
                    LOG.warn("Attempting fallback to CPU kernel");
                    selectKernelForPlatform("cpu");
                } else {
                    throw new KernelInitializationException(platform,
                            "Failed to initialize kernel and no CPU fallback available", e);
                }
            }
        } else {
            LOG.warnf("No available kernel for platform %s, falling back to CPU", platform);

            // Try fallback to CPU
            if (!"cpu".equals(platform)) {
                selectKernelForPlatform("cpu");
            } else {
                activeKernel = null;
                activePlatform = "cpu";
            }
        }
    }

    /**
     * Get the active kernel plugin.
     *
     * @return active kernel or null
     */
    public Optional<KernelPlugin> getActiveKernel() {
        return Optional.ofNullable(activeKernel);
    }

    /**
     * Get kernel for specific platform.
     *
     * @param platform platform
     * @return optional kernel
     */
    public Optional<KernelPlugin> getKernelForPlatform(String platform) {
        return Optional.ofNullable(kernels.get(platform));
    }

    /**
     * Execute operation on active kernel.
     *
     * @param operation operation to execute
     * @param context execution context
     * @param <T> result type
     * @return execution result
     * @throws KernelException if execution fails
     */
    public <T> KernelResult<T> execute(KernelOperation operation, KernelContext context)
            throws KernelException {

        if (activeKernel == null) {
            throw new KernelNotFoundException(activePlatform != null ?
                    activePlatform : "unknown", "No active kernel available");
        }

        KernelPluginState state = kernelStates.get(activePlatform);
        if (state != KernelPluginState.ACTIVE) {
            throw new KernelExecutionException(activePlatform, operation.getName(),
                    "Kernel is not in ACTIVE state: " + state);
        }

        long startTime = System.nanoTime();
        try {
            LOG.debugf("Executing operation %s on kernel %s",
                    operation.getName(), activeKernel.id());

            KernelResult<T> result = activeKernel.execute(operation, context);

            long durationMs = (System.nanoTime() - startTime) / 1_000_000;
            metrics.recordOperation(operation.getName(), durationMs, result.isSuccess());

            return result;

        } catch (KernelException e) {
            long durationMs = (System.nanoTime() - startTime) / 1_000_000;
            metrics.recordOperation(operation.getName(), durationMs, false);
            metrics.recordError("execution_failed", e);
            throw e;

        } catch (Exception e) {
            long durationMs = (System.nanoTime() - startTime) / 1_000_000;
            metrics.recordOperation(operation.getName(), durationMs, false);
            metrics.recordError("execution_error", e);
            throw new KernelExecutionException(activePlatform, operation.getName(),
                    "Execution failed", e);
        }
    }

    /**
     * Execute operation on active kernel (legacy method).
     *
     * @param operation operation name
     * @param params parameters
     * @return result
     * @throws KernelException if execution fails
     */
    public Object execute(String operation, Map<String, Object> params) throws KernelException {
        KernelOperation op = new KernelOperation(operation, params);
        KernelContext context = KernelContext.withParameters(params);
        KernelResult<Object> result = execute(op, context);
        return result.getData();
    }

    /**
     * Execute operation asynchronously.
     *
     * @param operation operation to execute
     * @param context execution context
     * @param <T> result type
     * @return completion stage with result
     */
    public <T> CompletionStage<KernelResult<T>> executeAsync(
            KernelOperation operation, KernelContext context) {

        if (activeKernel == null) {
            java.util.concurrent.CompletableFuture<KernelResult<T>> future =
                new java.util.concurrent.CompletableFuture<>();
            future.completeExceptionally(
                new KernelNotFoundException("No active kernel available"));
            return future;
        }

        return activeKernel.executeAsync(operation, context);
    }

    /**
     * Get all registered kernels.
     *
     * @return list of kernels
     */
    public List<KernelPlugin> getAllKernels() {
        return new ArrayList<>(kernels.values());
    }

    /**
     * Get current platform.
     *
     * @return current platform
     */
    public String getCurrentPlatform() {
        return activePlatform != null ? activePlatform : "unknown";
    }

    /**
     * Get health status of all kernels.
     *
     * @return health status map
     */
    public Map<String, KernelHealth> getHealthStatus() {
        Map<String, KernelHealth> status = new HashMap<>();

        for (Map.Entry<String, KernelPlugin> entry : kernels.entrySet()) {
            String platform = entry.getKey();
            KernelPlugin kernel = entry.getValue();

            try {
                status.put(platform, kernel.health());
            } catch (Exception e) {
                status.put(platform, KernelHealth.unhealthy(
                    "Health check failed: " + e.getMessage()));
            }
        }

        return status;
    }

    /**
     * Get kernel plugin manager statistics.
     *
     * @return statistics map
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("initialized", initialized.get());
        stats.put("active_platform", getCurrentPlatform());
        stats.put("active_kernel", activeKernel != null ? activeKernel.id() : "none");
        stats.put("total_kernels", kernels.size());
        stats.put("kernel_states", new HashMap<>(kernelStates));
        stats.put("metrics", metrics.toMap());

        List<Map<String, Object>> kernelDetails = kernels.values().stream()
                .map(k -> Map.<String, Object>of(
                        "id", k.id(),
                        "platform", k.platform(),
                        "available", k.isAvailable(),
                        "healthy", k.isHealthy(),
                        "state", kernelStates.getOrDefault(k.platform(), KernelPluginState.LOADED).toString()))
                .toList();
        stats.put("kernels", kernelDetails);

        return stats;
    }

    /**
     * Get kernel metrics.
     *
     * @return kernel metrics
     */
    public KernelMetrics getMetrics() {
        return metrics;
    }

    /**
     * Register kernel listener.
     *
     * @param listener listener to register
     */
    public void addListener(KernelListener listener) {
        listeners.add(listener);
    }

    /**
     * Unregister kernel listener.
     *
     * @param listener listener to unregister
     */
    public void removeListener(KernelListener listener) {
        listeners.remove(listener);
    }

    /**
     * Shutdown all kernels.
     */
    @jakarta.annotation.PreDestroy
    public void shutdown() {
        if (!initialized.get()) {
            return;
        }

        LOG.info("Shutting down kernel plugins...");
        metrics.recordEvent("shutdown_started");

        for (KernelPlugin kernel : kernels.values()) {
            try {
                LOG.debugf("Shutting down kernel: %s", kernel.id());
                kernel.shutdown();
                kernelStates.put(kernel.platform(), KernelPluginState.STOPPED);
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down kernel %s", kernel.id());
                metrics.recordError("shutdown_failed", e);
            }
        }

        kernels.clear();
        kernelStates.clear();
        activeKernel = null;
        activePlatform = null;
        initialized.set(false);

        metrics.recordEvent("shutdown_completed");
        LOG.info("Kernel plugin manager shutdown complete");
    }

    // ========================================================================
    // Notification Methods
    // ========================================================================

    private void notifyKernelRegistered(KernelPlugin kernel) {
        for (KernelListener listener : listeners) {
            try {
                listener.onKernelRegistered(kernel);
            } catch (Exception e) {
                LOG.errorf(e, "Listener notification failed for kernel registration");
            }
        }
    }

    private void notifyKernelActivated(KernelPlugin kernel) {
        for (KernelListener listener : listeners) {
            try {
                listener.onKernelActivated(kernel);
            } catch (Exception e) {
                LOG.errorf(e, "Listener notification failed for kernel activation");
            }
        }
    }

    // ========================================================================
    // Kernel Plugin State
    // ========================================================================

    /**
     * Kernel plugin lifecycle states.
     */
    public enum KernelPluginState {
        /** Kernel loaded but not validated */
        LOADED,

        /** Kernel is being validated */
        VALIDATING,

        /** Kernel validated successfully */
        VALIDATED,

        /** Kernel is being initialized */
        INITIALIZING,

        /** Kernel active and ready */
        ACTIVE,

        /** Kernel stopped */
        STOPPED,

        /** Kernel encountered an error */
        ERROR
    }

    // ========================================================================
    // Kernel Listener Interface
    // ========================================================================

    /**
     * Listener for kernel events.
     */
    public interface KernelListener {

        /**
         * Called when kernel is registered.
         *
         * @param kernel registered kernel
         */
        default void onKernelRegistered(KernelPlugin kernel) {
            // Default: no-op
        }

        /**
         * Called when kernel is activated.
         *
         * @param kernel activated kernel
         */
        default void onKernelActivated(KernelPlugin kernel) {
            // Default: no-op
        }

        /**
         * Called when kernel state changes.
         *
         * @param kernel kernel
         * @param oldState old state
         * @param newState new state
         */
        default void onKernelStateChanged(KernelPlugin kernel,
                                          KernelPluginState oldState,
                                          KernelPluginState newState) {
            // Default: no-op
        }
    }
}
