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

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.kernel.KernelPluginManager;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for kernel plugin producer.
 *
 * Verifies that:
 * - KernelPluginProducer is properly initialized
 * - KernelPluginManager is produced correctly
 * - KernelPluginIntegration is produced correctly
 * - CDI injection works correctly
 */
@QuarkusTest
public class KernelPluginProducerTest {

    @Inject
    KernelPluginProducer kernelPluginProducer;

    @Inject
    KernelPluginManager kernelPluginManager;

    @Inject
    KernelPluginIntegration kernelPluginIntegration;

    @Test
    public void testProducerInjected() {
        assertNotNull(kernelPluginProducer, "KernelPluginProducer should be injected");
    }

    @Test
    public void testProducerInitialized() {
        assertTrue(kernelPluginProducer.isInitialized(),
                "KernelPluginProducer should be initialized");
    }

    @Test
    public void testKernelPluginManagerProduced() {
        // Test direct injection
        assertNotNull(kernelPluginManager,
                "KernelPluginManager should be directly injected");

        // Test via producer
        KernelPluginManager manager = kernelPluginProducer.produceKernelPluginManager();
        assertNotNull(manager,
                "KernelPluginManager should be produced");

        // Verify same instance
        assertSame(kernelPluginManager, manager,
                "Should produce same KernelPluginManager instance");
    }

    @Test
    public void testKernelPluginIntegrationProduced() {
        // Test direct injection
        assertNotNull(kernelPluginIntegration,
                "KernelPluginIntegration should be directly injected");

        // Test via producer
        KernelPluginIntegration integration = kernelPluginProducer.produceKernelPluginIntegration();
        assertNotNull(integration,
                "KernelPluginIntegration should be produced");

        // Verify same instance
        assertSame(kernelPluginIntegration, integration,
                "Should produce same KernelPluginIntegration instance");
    }

    @Test
    public void testKernelPluginManagerAccess() {
        // Access via producer
        KernelPluginManager manager = kernelPluginProducer.getKernelManager();
        assertNotNull(manager, "Should get KernelPluginManager from producer");

        // Verify manager is functional
        assertNotNull(manager.getAllKernels(),
                "KernelPluginManager should be functional");
    }

    @Test
    public void testKernelPluginIntegrationAccess() {
        // Access via producer
        KernelPluginIntegration integration = kernelPluginProducer.getKernelIntegration();
        assertNotNull(integration, "Should get KernelPluginIntegration from producer");

        // Verify integration is functional
        assertTrue(integration.isInitialized(),
                "KernelPluginIntegration should be initialized");
    }

    @Test
    public void testKernelPluginHealth() {
        // Get kernel plugin manager
        KernelPluginManager manager = kernelPluginProducer.getKernelManager();
        assertNotNull(manager, "KernelPluginManager should be available");

        // Get health status
        var health = manager.getHealthStatus();
        assertNotNull(health, "Health status should be available");
    }

    @Test
    public void testKernelPluginStats() {
        // Get kernel plugin manager
        KernelPluginManager manager = kernelPluginProducer.getKernelManager();
        assertNotNull(manager, "KernelPluginManager should be available");

        // Get stats
        var stats = manager.getStats();
        assertNotNull(stats, "Stats should be available");

        // Verify expected keys
        assertTrue(stats.containsKey("initialized"),
                "Stats should contain 'initialized'");
        assertTrue(stats.containsKey("active_platform"),
                "Stats should contain 'active_platform'");
        assertTrue(stats.containsKey("total_kernels"),
                "Stats should contain 'total_kernels'");
    }

    @Test
    public void testProducerShutdown() {
        // Verify producer can be shut down
        assertDoesNotThrow(() -> kernelPluginProducer.shutdown(),
                "Producer shutdown should not throw exception");
    }
}
