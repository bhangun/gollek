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
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.kernel.KernelPluginManager;

/**
 * CDI Producer for kernel plugin system components.
 *
 * <p>
 * Provides dependency injection support for:
 * <ul>
 * <li>KernelPluginManager - Main kernel plugin manager</li>
 * <li>KernelPluginIntegration - Integration bridge</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * 
 * <pre>
 * {
 *     &#64;code
 *     &#64;Inject
 *     KernelPluginManager kernelManager;
 *
 *     @Inject
 *     KernelPluginIntegration kernelIntegration;
 * }
 * </pre>
 *
 * @since 2.0.0
 */
@ApplicationScoped
public class KernelPluginProducer {

    private static final Logger LOG = Logger.getLogger(KernelPluginProducer.class);

    private volatile KernelPluginIntegration kernelIntegration;
    private volatile KernelPluginManager kernelPluginManager;
    private volatile boolean initialized = false;

    /**
     * Initialize kernel plugin integration.
     */
    public synchronized void initialize() {
        if (initialized) {
            return;
        }

        LOG.info("Initializing Kernel Plugin Producer...");

        try {
            // Create kernel plugin integration
            kernelIntegration = new KernelPluginIntegration();
            kernelIntegration.initialize();

            // Get kernel plugin manager
            kernelPluginManager = kernelIntegration.getKernelPluginManager();

            initialized = true;

            LOG.infof("Kernel Plugin Producer initialized successfully");
            LOG.infof("  Active platform: %s", kernelPluginManager.getCurrentPlatform());
            LOG.infof("  Total kernels: %d", kernelPluginManager.getAllKernels().size());
            LOG.infof("  Health status: %s",
                    kernelPluginManager.getActiveKernel()
                            .map(k -> k.isHealthy() ? "✓ Healthy" : "✗ Unhealthy")
                            .orElse("✗ No active kernel"));

        } catch (Exception e) {
            LOG.error("Failed to initialize Kernel Plugin Producer", e);
            throw new RuntimeException("Kernel Plugin Producer initialization failed", e);
        }
    }

    /**
     * Produce KernelPluginIntegration for CDI injection.
     *
     * @return kernel plugin integration
     */
    @Produces
    @Singleton
    public KernelPluginIntegration produceKernelPluginIntegration() {
        if (!initialized) {
            initialize();
        }
        return kernelIntegration;
    }

    /**
     * Produce KernelPluginManager for CDI injection.
     *
     * @return kernel plugin manager
     */
    @Produces
    @Singleton
    public KernelPluginManager produceKernelPluginManager() {
        if (!initialized) {
            initialize();
        }
        return kernelPluginManager;
    }

    /**
     * Check if producer is initialized.
     *
     * @return true if initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get kernel plugin integration.
     *
     * @return kernel plugin integration
     */
    public KernelPluginIntegration getKernelIntegration() {
        return kernelIntegration;
    }

    /**
     * Get kernel plugin manager.
     *
     * @return kernel plugin manager
     */
    public KernelPluginManager getKernelManager() {
        return kernelPluginManager;
    }

    /**
     * Shutdown kernel plugin producer.
     */
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down Kernel Plugin Producer...");

        try {
            if (kernelIntegration != null) {
                kernelIntegration.shutdown();
            }
        } catch (Exception e) {
            LOG.error("Error shutting down Kernel Plugin Producer", e);
        }

        initialized = false;
        kernelIntegration = null;
        kernelPluginManager = null;

        LOG.info("Kernel Plugin Producer shutdown complete");
    }
}
