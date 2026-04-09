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

package tech.kayys.gollek.engine.kernel;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.kernel.KernelPlugin;
import tech.kayys.gollek.plugin.kernel.KernelPluginManager;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.plugin.PluginContext;

/**
 * Kernel plugin system integration with the main plugin manager.
 *
 * <p>
 * Bridges the enhanced kernel plugin system with the core plugin
 * infrastructure,
 * providing:
 * </p>
 * <ul>
 * <li>Standalone kernel plugin manager initialization</li>
 * <li>Kernel plugin adapter for GollekPlugin interface</li>
 * <li>Integration with PluginManager lifecycle</li>
 * <li>Health monitoring and metrics exposure</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * @Inject
 * KernelPluginIntegration kernelIntegration;
 *
 * // Initialize
 * kernelIntegration.initialize();
 *
 * // Get kernel plugin manager
 * KernelPluginManager kernelManager = kernelIntegration.getKernelPluginManager();
 *
 * // Execute operation
 * KernelOperation operation = KernelOperation.builder()
 *         .name("gemm")
 *         .parameter("m", 1024)
 *         .build();
 *
 * KernelResult<Matrix> result = kernelManager.execute(operation, KernelContext.empty());
 * }</pre>
 *
 * @since 2.1.0
 * @version 2.0.0
 */
@ApplicationScoped
public class KernelPluginIntegration {

    private static final Logger LOG = Logger.getLogger(KernelPluginIntegration.class);

    @Inject
    Instance<KernelPlugin> kernelPluginInstances;

    private volatile KernelPluginManager kernelPluginManager;
    private volatile boolean initialized = false;

    /**
     * Create kernel plugin integration.
     */
    public KernelPluginIntegration() {
        // Default constructor
    }

    /**
     * Initialize kernel plugin integration.
     */
    public void initialize() {
        if (initialized) {
            LOG.warn("Kernel plugin integration already initialized");
            return;
        }

        LOG.info("Initializing kernel plugin integration...");

        try {
            // Create kernel plugin manager
            kernelPluginManager = new KernelPluginManager();

            // Inject kernel plugin instances if available
            if (kernelPluginInstances != null) {
                kernelPluginInstances.forEach(kernel -> {
                    try {
                        kernelPluginManager.register(kernel);
                        LOG.infof("Registered kernel plugin: %s", kernel.id());
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to register kernel plugin: %s", kernel.id());
                    }
                });
            }

            // Initialize kernel plugin manager
            kernelPluginManager.initialize();

            initialized = true;
            LOG.info("Kernel plugin integration initialized successfully");
            LOG.infof("  Active platform: %s", kernelPluginManager.getCurrentPlatform());
            LOG.infof("  Total kernels: %d", kernelPluginManager.getAllKernels().size());

        } catch (Exception e) {
            LOG.error("Failed to initialize kernel plugin integration", e);
            throw new RuntimeException("Kernel plugin integration failed", e);
        }
    }

    /**
     * Get the kernel plugin manager.
     *
     * @return kernel plugin manager
     */
    public KernelPluginManager getKernelPluginManager() {
        if (!initialized) {
            throw new IllegalStateException("Kernel plugin integration not initialized");
        }
        return kernelPluginManager;
    }

    /**
     * Check if kernel plugin system is initialized.
     *
     * @return true if initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get kernel plugin count.
     *
     * @return number of registered kernels
     */
    public int getKernelCount() {
        if (!initialized) {
            return 0;
        }
        return kernelPluginManager.getAllKernels().size();
    }

    /**
     * Get active kernel platform.
     *
     * @return active platform or "unknown"
     */
    public String getActivePlatform() {
        if (!initialized) {
            return "unknown";
        }
        return kernelPluginManager.getCurrentPlatform();
    }

    /**
     * Get kernel plugin health status.
     *
     * @return true if healthy
     */
    public boolean isHealthy() {
        if (!initialized) {
            return false;
        }
        return kernelPluginManager.getActiveKernel()
                .map(KernelPlugin::isHealthy)
                .orElse(false);
    }

    /**
     * Get kernel plugin metrics summary.
     *
     * @return metrics summary map
     */
    public java.util.Map<String, Object> getMetricsSummary() {
        if (!initialized) {
            return java.util.Map.of();
        }
        return kernelPluginManager.getStats();
    }

    /**
     * Shutdown kernel plugin integration.
     */
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down kernel plugin integration...");

        try {
            if (kernelPluginManager != null) {
                kernelPluginManager.shutdown();
            }
        } catch (Exception e) {
            LOG.error("Error shutting down kernel plugin manager", e);
        }

        initialized = false;
        LOG.info("Kernel plugin integration shutdown complete");
    }

    // ========================================================================
    // Kernel Plugin Adapter
    // ========================================================================

    /**
     * Adapter that wraps KernelPluginManager as a GollekPlugin.
     * Allows kernel plugin system to integrate with main plugin infrastructure.
     */
    public static class KernelPluginAdapter implements GollekPlugin {

        private final KernelPluginManager kernelManager;

        public KernelPluginAdapter(KernelPluginManager kernelManager) {
            this.kernelManager = kernelManager;
        }

        @Override
        public String id() {
            return "kernel-plugin-manager";
        }

        @Override
        public int order() {
            return 10; // High priority - kernels initialize first
        }

        @Override
        public String version() {
            return "2.0.0";
        }

        @Override
        public void initialize(PluginContext context) {
            LOG.info("Kernel plugin adapter initializing...");
            if (kernelManager != null) {
                kernelManager.initialize();
            }
        }

        @Override
        public void start() {
            LOG.infof("Kernel plugin adapter started. Active platform: %s",
                    kernelManager != null ? kernelManager.getCurrentPlatform() : "none");
        }

        @Override
        public void stop() {
            LOG.info("Kernel plugin adapter stopping...");
        }

        @Override
        public void shutdown() {
            LOG.info("Kernel plugin adapter shutting down...");
            if (kernelManager != null) {
                kernelManager.shutdown();
            }
        }

        @Override
        public boolean isHealthy() {
            if (kernelManager == null) {
                return false;
            }
            return kernelManager.getActiveKernel()
                    .map(KernelPlugin::isHealthy)
                    .orElse(false);
        }

        @Override
        public PluginMetadata metadata() {
            return new PluginMetadata(
                    id(),
                    version(),
                    KernelPluginManager.class.getName(),
                    order());
        }
    }
}
