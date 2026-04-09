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

package tech.kayys.gollek.engine.runner;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.List;

/**
 * CDI Producer for runner plugin system components.
 *
 * <p>Provides dependency injection support for:
 * <ul>
 *   <li>RunnerPluginManager - Main runner plugin manager</li>
 *   <li>RunnerPluginIntegration - Integration bridge</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * @Inject
 * RunnerPluginManager runnerManager;
 *
 * @Inject
 * RunnerPluginIntegration runnerIntegration;
 * }</pre>
 *
 * @since 2.0.0
 */
@ApplicationScoped
public class RunnerPluginProducer {

    private static final Logger LOG = Logger.getLogger(RunnerPluginProducer.class);

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    private volatile RunnerPluginManager runnerPluginManager;
    private volatile RunnerPluginIntegration runnerIntegration;
    private volatile boolean initialized = false;

    /**
     * Initialize runner plugin integration.
     */
    public synchronized void initialize() {
        if (initialized) {
            return;
        }

        LOG.info("Initializing Runner Plugin Producer...");

        try {
            // Create runner plugin manager
            runnerPluginManager = RunnerPluginManager.getInstance();

            // Discover and register runner plugins via CDI
            if (runnerPluginInstances != null) {
                int count = 0;
                for (RunnerPlugin plugin : runnerPluginInstances) {
                    try {
                        runnerPluginManager.register(plugin);
                        LOG.infof("Registered runner plugin: %s (version %s)",
                                plugin.id(), plugin.version());
                        count++;
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to register runner plugin: %s", plugin.id());
                    }
                }
                LOG.infof("Registered %d runner plugins", count);
            }

            // Create runner plugin integration
            runnerIntegration = new RunnerPluginIntegration(runnerPluginManager);

            initialized = true;

            LOG.infof("Runner Plugin Producer initialized successfully");
            LOG.infof("  Total runners: %d", runnerPluginManager.getAllPlugins().size());
            LOG.infof("  Available runners: %d", runnerPluginManager.getAvailablePlugins().size());

        } catch (Exception e) {
            LOG.error("Failed to initialize Runner Plugin Producer", e);
            throw new RuntimeException("Runner Plugin Producer initialization failed", e);
        }
    }

    /**
     * Produce RunnerPluginManager for CDI injection.
     *
     * @return runner plugin manager
     */
    @jakarta.enterprise.inject.Produces
    @jakarta.inject.Singleton
    public RunnerPluginManager produceRunnerPluginManager() {
        if (!initialized) {
            initialize();
        }
        return runnerPluginManager;
    }

    /**
     * Produce RunnerPluginIntegration for CDI injection.
     *
     * @return runner plugin integration
     */
    @jakarta.enterprise.inject.Produces
    @jakarta.inject.Singleton
    public RunnerPluginIntegration produceRunnerPluginIntegration() {
        if (!initialized) {
            initialize();
        }
        return runnerIntegration;
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
     * Get runner plugin manager.
     *
     * @return runner plugin manager
     */
    public RunnerPluginManager getRunnerManager() {
        return runnerPluginManager;
    }

    /**
     * Get runner plugin integration.
     *
     * @return runner plugin integration
     */
    public RunnerPluginIntegration getRunnerIntegration() {
        return runnerIntegration;
    }

    /**
     * Get all registered runner plugins.
     *
     * @return list of runner plugins
     */
    public List<RunnerPlugin> getAllRunners() {
        if (!initialized) {
            return List.of();
        }
        return runnerPluginManager.getAllPlugins();
    }

    /**
     * Get available runner plugins.
     *
     * @return list of available runner plugins
     */
    public List<RunnerPlugin> getAvailableRunners() {
        if (!initialized) {
            return List.of();
        }
        return runnerPluginManager.getAvailablePlugins();
    }

    /**
     * Shutdown runner plugin producer.
     */
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down Runner Plugin Producer...");

        try {
            if (runnerPluginManager != null) {
                runnerPluginManager.shutdown();
            }
        } catch (Exception e) {
            LOG.error("Error shutting down Runner Plugin Producer", e);
        }

        initialized = false;
        runnerPluginManager = null;
        runnerIntegration = null;

        LOG.info("Runner Plugin Producer shutdown complete");
    }
}
