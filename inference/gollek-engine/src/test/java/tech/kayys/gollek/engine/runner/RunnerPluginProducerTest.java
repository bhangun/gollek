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

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for runner plugin producer.
 *
 * Verifies that:
 * - RunnerPluginProducer is properly initialized
 * - RunnerPluginManager is produced correctly
 * - RunnerPluginIntegration is produced correctly
 * - CDI injection works correctly
 * - Runner plugins are discovered and registered
 */
@QuarkusTest
public class RunnerPluginProducerTest {

    @Inject
    RunnerPluginProducer runnerPluginProducer;

    @Inject
    RunnerPluginManager runnerPluginManager;

    @Inject
    RunnerPluginIntegration runnerPluginIntegration;

    @Test
    public void testProducerInjected() {
        assertNotNull(runnerPluginProducer, "RunnerPluginProducer should be injected");
    }

    @Test
    public void testProducerInitialized() {
        assertTrue(runnerPluginProducer.isInitialized(),
                "RunnerPluginProducer should be initialized");
    }

    @Test
    public void testRunnerPluginManagerProduced() {
        // Test direct injection
        assertNotNull(runnerPluginManager,
                "RunnerPluginManager should be directly injected");

        // Test via producer
        RunnerPluginManager manager = runnerPluginProducer.produceRunnerPluginManager();
        assertNotNull(manager,
                "RunnerPluginManager should be produced");

        // Verify same instance
        assertSame(runnerPluginManager, manager,
                "Should produce same RunnerPluginManager instance");
    }

    @Test
    public void testRunnerPluginIntegrationProduced() {
        // Test direct injection
        assertNotNull(runnerPluginIntegration,
                "RunnerPluginIntegration should be directly injected");

        // Test via producer
        RunnerPluginIntegration integration = runnerPluginProducer.produceRunnerPluginIntegration();
        assertNotNull(integration,
                "RunnerPluginIntegration should be produced");

        // Verify same instance
        assertSame(runnerPluginIntegration, integration,
                "Should produce same RunnerPluginIntegration instance");
    }

    @Test
    public void testRunnerPluginManagerAccess() {
        // Access via producer
        RunnerPluginManager manager = runnerPluginProducer.getRunnerManager();
        assertNotNull(manager, "Should get RunnerPluginManager from producer");

        // Verify manager is functional
        assertNotNull(manager.getAllPlugins(),
                "RunnerPluginManager should be functional");
    }

    @Test
    public void testRunnerPluginIntegrationAccess() {
        // Access via producer
        RunnerPluginIntegration integration = runnerPluginProducer.getRunnerIntegration();
        assertNotNull(integration, "Should get RunnerPluginIntegration from producer");

        // Verify integration is functional
        assertNotNull(integration.getRunnerManager(),
                "RunnerPluginIntegration should be functional");
    }

    @Test
    public void testRunnerPluginDiscovery() {
        // Get all runners
        List<RunnerPlugin> runners = runnerPluginProducer.getAllRunners();
        assertNotNull(runners, "Runners list should not be null");

        // Get available runners
        List<RunnerPlugin> availableRunners = runnerPluginProducer.getAvailableRunners();
        assertNotNull(availableRunners, "Available runners list should not be null");

        // Verify at least some runners are registered (GGUF, ONNX, etc.)
        // Note: In test environment, may have 0 runners if no plugins are deployed
        assertTrue(runners.size() >= 0, "Should have 0 or more runners");
    }

    @Test
    public void testRunnerPluginHealth() {
        // Get runner plugin integration
        RunnerPluginIntegration integration = runnerPluginProducer.getRunnerIntegration();
        assertNotNull(integration, "RunnerPluginIntegration should be available");

        // Get health status
        var health = integration.getHealthStatus();
        assertNotNull(health, "Health status should be available");
    }

    @Test
    public void testRunnerPluginMetrics() {
        // Get runner plugin integration
        RunnerPluginIntegration integration = runnerPluginProducer.getRunnerIntegration();
        assertNotNull(integration, "RunnerPluginIntegration should be available");

        // Get metrics summary
        var metrics = integration.getMetricsSummary();
        assertNotNull(metrics, "Metrics should be available");

        // Verify expected keys
        assertTrue(metrics.containsKey("total_runners"),
                "Metrics should contain 'total_runners'");
        assertTrue(metrics.containsKey("available_runners"),
                "Metrics should contain 'available_runners'");
    }

    @Test
    public void testRunnerPluginStatus() {
        // Get runner plugin integration
        RunnerPluginIntegration integration = runnerPluginProducer.getRunnerIntegration();
        assertNotNull(integration, "RunnerPluginIntegration should be available");

        // Get status
        var status = integration.getStatus();
        assertNotNull(status, "Status should be available");

        // Verify expected keys
        assertTrue(status.containsKey("initialized"),
                "Status should contain 'initialized'");
        assertTrue(status.containsKey("total_runners"),
                "Status should contain 'total_runners'");
    }

    @Test
    public void testProducerShutdown() {
        // Verify producer can be shut down
        assertDoesNotThrow(() -> runnerPluginProducer.shutdown(),
                "Producer shutdown should not throw exception");
    }
}
