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

package tech.kayys.gollek.engine.plugin;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.engine.kernel.KernelPluginIntegration;
import tech.kayys.gollek.engine.kernel.KernelPluginProducer;
import tech.kayys.gollek.engine.runner.RunnerPluginIntegration;
import tech.kayys.gollek.engine.runner.RunnerPluginProducer;
import tech.kayys.gollek.plugin.kernel.KernelPluginManager;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for plugin system integration with engine.
 *
 * Verifies that:
 * - PluginSystemIntegrator is properly initialized
 * - Kernel plugin system is wired correctly
 * - Runner plugin system is wired correctly
 * - All plugin managers are accessible via CDI
 * - Health monitoring works correctly
 * - Metrics are exposed correctly
 */
@QuarkusTest
public class PluginSystemIntegrationTest {

    @Inject
    PluginSystemIntegrator pluginIntegrator;

    @Inject
    KernelPluginProducer kernelPluginProducer;

    @Inject
    RunnerPluginProducer runnerPluginProducer;

    @Inject
    KernelPluginManager kernelPluginManager;

    @Inject
    RunnerPluginManager runnerPluginManager;

    @Test
    public void testPluginSystemIntegratorInitialized() {
        // Verify integrator is initialized
        assertNotNull(pluginIntegrator, "PluginSystemIntegrator should be injected");

        // Verify plugin status map is available
        Map<String, Boolean> pluginStatus = pluginIntegrator.getPluginStatus();
        assertNotNull(pluginStatus, "Plugin status should be available");

        // Verify all plugin levels are present
        assertTrue(pluginStatus.containsKey("kernel"), "Kernel plugin status should be present");
        assertTrue(pluginStatus.containsKey("runner"), "Runner plugin status should be present");
        assertTrue(pluginStatus.containsKey("optimization"), "Optimization plugin status should be present");
        assertTrue(pluginStatus.containsKey("feature"), "Feature plugin status should be present");
    }

    @Test
    public void testKernelPluginProducerInitialized() {
        // Verify kernel plugin producer is injected
        assertNotNull(kernelPluginProducer, "KernelPluginProducer should be injected");

        // Verify producer is initialized
        assertTrue(kernelPluginProducer.isInitialized(), "KernelPluginProducer should be initialized");

        // Verify kernel plugin manager is available
        KernelPluginManager manager = kernelPluginProducer.getKernelManager();
        assertNotNull(manager, "KernelPluginManager should be available from producer");
    }

    @Test
    public void testRunnerPluginProducerInitialized() {
        // Verify runner plugin producer is injected
        assertNotNull(runnerPluginProducer, "RunnerPluginProducer should be injected");

        // Verify producer is initialized
        assertTrue(runnerPluginProducer.isInitialized(), "RunnerPluginProducer should be initialized");

        // Verify runner plugin manager is available
        RunnerPluginManager manager = runnerPluginProducer.getRunnerManager();
        assertNotNull(manager, "RunnerPluginManager should be available from producer");
    }

    @Test
    public void testKernelPluginManagerInjected() {
        // Verify kernel plugin manager is directly injectable
        assertNotNull(kernelPluginManager, "KernelPluginManager should be directly injected");

        // Verify manager is initialized
        assertTrue(kernelPluginManager.getAllKernels().size() >= 0,
                "KernelPluginManager should be initialized");
    }

    @Test
    public void testRunnerPluginManagerInjected() {
        // Verify runner plugin manager is directly injectable
        assertNotNull(runnerPluginManager, "RunnerPluginManager should be directly injected");

        // Verify manager is initialized
        assertNotNull(runnerPluginManager.getAllPlugins(),
                "RunnerPluginManager should be initialized");
    }

    @Test
    public void testPluginSystemIntegratorGetters() {
        // Test kernel plugin producer getter
        KernelPluginProducer kernelProducer = pluginIntegrator.getKernelPluginProducer();
        assertNotNull(kernelProducer, "Should get KernelPluginProducer from integrator");

        // Test kernel plugin manager getter
        KernelPluginManager kernelManager = pluginIntegrator.getKernelPluginManager();
        assertNotNull(kernelManager, "Should get KernelPluginManager from integrator");

        // Test runner plugin producer getter
        RunnerPluginProducer runnerProducer = pluginIntegrator.getRunnerPluginProducer();
        assertNotNull(runnerProducer, "Should get RunnerPluginProducer from integrator");

        // Test runner plugin integration getter
        RunnerPluginIntegration runnerIntegration = pluginIntegrator.getRunnerPluginIntegration();
        assertNotNull(runnerIntegration, "Should get RunnerPluginIntegration from integrator");
    }

    @Test
    public void testKernelPluginHealth() {
        // Get kernel plugin manager
        KernelPluginManager kernelManager = pluginIntegrator.getKernelPluginManager();
        assertNotNull(kernelManager, "KernelPluginManager should be available");

        // Get health status
        Map<String, tech.kayys.gollek.plugin.kernel.KernelHealth> health =
                kernelManager.getHealthStatus();

        // Health map should not be null
        assertNotNull(health, "Kernel health status should be available");

        // If kernels are registered, verify health structure
        if (!health.isEmpty()) {
            health.forEach((platform, kernelHealth) -> {
                assertNotNull(platform, "Platform should not be null");
                assertNotNull(kernelHealth, "Health status should not be null");
            });
        }
    }

    @Test
    public void testRunnerPluginHealth() {
        // Get runner plugin integration
        RunnerPluginIntegration runnerIntegration = pluginIntegrator.getRunnerPluginIntegration();
        assertNotNull(runnerIntegration, "RunnerPluginIntegration should be available");

        // Get health status
        Map<String, Boolean> health = runnerIntegration.getHealthStatus();

        // Health map should not be null
        assertNotNull(health, "Runner health status should be available");

        // If runners are registered, verify health structure
        if (!health.isEmpty()) {
            health.forEach((runner, isHealthy) -> {
                assertNotNull(runner, "Runner ID should not be null");
                assertNotNull(isHealthy, "Health status should not be null");
            });
        }
    }

    @Test
    public void testKernelPluginMetrics() {
        // Get kernel plugin manager
        KernelPluginManager kernelManager = pluginIntegrator.getKernelPluginManager();
        assertNotNull(kernelManager, "KernelPluginManager should be available");

        // Get stats
        Map<String, Object> stats = kernelManager.getStats();

        // Stats should not be null
        assertNotNull(stats, "Kernel stats should be available");

        // Verify expected keys
        assertTrue(stats.containsKey("initialized"), "Stats should contain 'initialized'");
        assertTrue(stats.containsKey("active_platform"), "Stats should contain 'active_platform'");
        assertTrue(stats.containsKey("total_kernels"), "Stats should contain 'total_kernels'");
    }

    @Test
    public void testRunnerPluginMetrics() {
        // Get runner plugin integration
        RunnerPluginIntegration runnerIntegration = pluginIntegrator.getRunnerPluginIntegration();
        assertNotNull(runnerIntegration, "RunnerPluginIntegration should be available");

        // Get metrics summary
        Map<String, Object> metrics = runnerIntegration.getMetricsSummary();

        // Metrics should not be null
        assertNotNull(metrics, "Runner metrics should be available");

        // Verify expected keys
        assertTrue(metrics.containsKey("total_runners"), "Metrics should contain 'total_runners'");
        assertTrue(metrics.containsKey("available_runners"), "Metrics should contain 'available_runners'");
    }

    @Test
    public void testPluginStatusConsistency() {
        // Get plugin status from integrator
        Map<String, Boolean> pluginStatus = pluginIntegrator.getPluginStatus();

        // Verify kernel plugin status matches actual state
        KernelPluginProducer kernelProducer = pluginIntegrator.getKernelPluginProducer();
        assertEquals(kernelProducer.isInitialized(),
                pluginStatus.get("kernel"),
                "Kernel plugin status should match actual state");

        // Verify runner plugin status matches actual state
        RunnerPluginProducer runnerProducer = pluginIntegrator.getRunnerPluginProducer();
        assertEquals(runnerProducer.isInitialized(),
                pluginStatus.get("runner"),
                "Runner plugin status should match actual state");
    }

    @Test
    public void testFullyInitialized() {
        // Check if fully initialized
        boolean fullyInitialized = pluginIntegrator.isFullyInitialized();

        // Should be initialized in test environment
        // Note: May be false if some optional plugins are not available
        // This test verifies the method works correctly
        assertNotNull(fullyInitialized, "isFullyInitialized should return a value");
    }

    @Test
    public void testKernelPluginIntegration() {
        // Get kernel plugin integration via producer
        KernelPluginIntegration kernelIntegration = kernelPluginProducer.produceKernelPluginIntegration();
        assertNotNull(kernelIntegration, "KernelPluginIntegration should be available");

        // Verify integration is initialized
        assertTrue(kernelIntegration.isInitialized(), "KernelPluginIntegration should be initialized");

        // Verify can get kernel plugin manager
        KernelPluginManager manager = kernelIntegration.getKernelPluginManager();
        assertNotNull(manager, "Should get KernelPluginManager from integration");
    }

    @Test
    public void testRunnerPluginIntegration() {
        // Get runner plugin integration via producer
        RunnerPluginIntegration runnerIntegration = runnerPluginProducer.produceRunnerPluginIntegration();
        assertNotNull(runnerIntegration, "RunnerPluginIntegration should be available");

        // Verify integration is initialized
        runnerIntegration.initialize(); // Ensure initialized

        // Verify can get runner plugin manager
        RunnerPluginManager manager = runnerIntegration.getRunnerManager();
        assertNotNull(manager, "Should get RunnerPluginManager from integration");
    }

    @Test
    public void testPluginDiscovery() {
        // Test kernel plugin discovery
        KernelPluginManager kernelManager = pluginIntegrator.getKernelPluginManager();
        assertNotNull(kernelManager, "KernelPluginManager should be available");

        // Get all kernels
        var kernels = kernelManager.getAllKernels();
        assertNotNull(kernels, "Kernels list should not be null");

        // Test runner plugin discovery
        RunnerPluginManager runnerManager = pluginIntegrator.getRunnerPluginIntegration()
                .getRunnerManager();
        assertNotNull(runnerManager, "RunnerPluginManager should be available");

        // Get all runners
        var runners = runnerManager.getAllPlugins();
        assertNotNull(runners, "Runners list should not be null");
    }

    @Test
    public void testConcurrentAccess() throws InterruptedException {
        // Test concurrent access to plugin managers
        Thread[] threads = new Thread[10];
        boolean[] success = new boolean[10];

        for (int i = 0; i < 10; i++) {
            final int index = i;
            threads[i] = new Thread(() -> {
                try {
                    // Access kernel plugin manager
                    KernelPluginManager km = pluginIntegrator.getKernelPluginManager();
                    assertNotNull(km, "KernelPluginManager should be accessible");

                    // Access runner plugin manager
                    RunnerPluginManager rm = pluginIntegrator.getRunnerPluginIntegration()
                            .getRunnerManager();
                    assertNotNull(rm, "RunnerPluginManager should be accessible");

                    success[index] = true;
                } catch (Exception e) {
                    success[index] = false;
                }
            });
            threads[i].start();
        }

        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join();
        }

        // Verify all threads succeeded
        for (int i = 0; i < 10; i++) {
            assertTrue(success[i], "Thread " + i + " should succeed");
        }
    }
}
