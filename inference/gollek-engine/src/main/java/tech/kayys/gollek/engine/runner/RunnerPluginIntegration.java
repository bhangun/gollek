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

import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.Map;

/**
 * Runner plugin system integration bridge.
 *
 * <p>Provides integration between runner plugin system and engine:
 * <ul>
 *   <li>Health monitoring</li>
 *   <li>Metrics exposure</li>
 *   <li>Plugin status reporting</li>
 * </ul>
 *
 * @since 2.0.0
 */
@jakarta.enterprise.context.ApplicationScoped
public class RunnerPluginIntegration {

    private static final Logger LOG = Logger.getLogger(RunnerPluginIntegration.class);

    private final RunnerPluginManager runnerManager;

    /**
     * Create runner plugin integration.
     *
     * @param runnerManager runner plugin manager
     */
    public RunnerPluginIntegration(RunnerPluginManager runnerManager) {
        this.runnerManager = runnerManager;
    }

    /**
     * Initialize runner plugin integration.
     */
    public void initialize() {
        LOG.info("Initializing runner plugin integration...");

        if (runnerManager == null) {
            LOG.warn("Runner plugin manager is null");
            return;
        }

        LOG.infof("Runner plugin integration initialized");
        LOG.infof("  Total runners: %d", runnerManager.getAllPlugins().size());
        LOG.infof("  Available runners: %d", runnerManager.getAvailablePlugins().size());
    }

    /**
     * Get runner plugin manager.
     *
     * @return runner plugin manager
     */
    public RunnerPluginManager getRunnerManager() {
        return runnerManager;
    }

    /**
     * Check if runner plugin system is initialized.
     *
     * @return true if initialized
     */
    public boolean isInitialized() {
        return runnerManager != null;
    }

    /**
     * Get runner plugin health status.
     *
     * @return health status map
     */
    public Map<String, Boolean> getHealthStatus() {
        if (runnerManager == null) {
            return Map.of();
        }
        return runnerManager.getHealthStatus();
    }

    /**
     * Check if all runner plugins are healthy.
     *
     * @return true if healthy
     */
    public boolean isHealthy() {
        if (runnerManager == null) {
            return false;
        }
        return runnerManager.getHealthStatus().values().stream().allMatch(Boolean::booleanValue);
    }

    /**
     * Get runner plugin metrics summary.
     *
     * @return metrics summary map
     */
    public Map<String, Object> getMetricsSummary() {
        if (runnerManager == null) {
            return Map.of();
        }

        return Map.of(
            "total_runners", runnerManager.getAllPlugins().size(),
            "available_runners", runnerManager.getAvailablePlugins().size(),
            "healthy_runners", (int) runnerManager.getHealthStatus().values().stream().filter(Boolean::booleanValue).count()
        );
    }

    /**
     * Get runner plugin status.
     *
     * @return status map
     */
    public Map<String, Object> getStatus() {
        if (runnerManager == null) {
            return Map.of("initialized", false);
        }

        return Map.of(
            "initialized", true,
            "total_runners", runnerManager.getAllPlugins().size(),
            "available_runners", runnerManager.getAvailablePlugins().size(),
            "health_status", getHealthStatus()
        );
    }

    /**
     * Shutdown runner plugin integration.
     */
    public void shutdown() {
        LOG.info("Shutting down runner plugin integration...");
        // RunnerPluginManager handles shutdown
    }
}
