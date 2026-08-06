/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.kernel;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test kernel platform detection.
 */
public class KernelPlatformDetectorTest {

    @Test
    public void testDetectorInstance() {
        KernelPlatformDetector detector = KernelPlatformDetector.getInstance();
        assertNotNull(detector, "Detector instance should be created");
    }

    @Test
    public void testDetectPlatform() {
        KernelPlatform platform = KernelPlatformDetector.detect();
        assertNotNull(platform, "Platform should be detected");
        
        // Should always return a platform (at minimum CPU)
        assertTrue(platform != null, "Platform should not be null");
    }

    @Test
    public void testGetAvailablePlatforms() {
        var platforms = KernelPlatformDetector.getAvailablePlatforms();
        assertNotNull(platforms, "Available platforms list should not be null");
        
        // Should always have at least CPU
        assertFalse(platforms.isEmpty(), "Should have at least CPU platform");
        assertTrue(platforms.contains(KernelPlatform.CPU), "Should always have CPU");
    }

    @Test
    public void testCpuAlwaysAvailable() {
        boolean isCpuAvailable = KernelPlatformDetector.isPlatformAvailable(KernelPlatform.CPU);
        assertTrue(isCpuAvailable, "CPU should always be available");
    }

    @Test
    public void testForceCpuProperty() {
        // Test CPU force property
        System.setProperty("gollek.kernel.force.cpu", "true");
        KernelPlatform platform = KernelPlatformDetector.detect();
        assertEquals(KernelPlatform.CPU, platform, "Should force CPU when property is set");
        
        // Clean up
        System.clearProperty("gollek.kernel.force.cpu");
    }

    @Test
    public void testForcePlatformProperty() {
        // Test platform force property
        System.setProperty("gollek.kernel.platform", "cpu");
        KernelPlatform platform = KernelPlatformDetector.detect();
        assertEquals(KernelPlatform.CPU, platform, "Should force platform when property is set");
        
        // Clean up
        System.clearProperty("gollek.kernel.platform");
    }

    @Test
    public void testPlatformEnum() {
        // Test all platform enum values
        for (KernelPlatform platform : KernelPlatform.values()) {
            assertNotNull(platform.getDisplayName(), "Display name should not be null for " + platform);
            assertNotNull(platform.getDescription(), "Description should not be null for " + platform);
            
            if (platform == KernelPlatform.CPU) {
                assertTrue(platform.isCpu(), "CPU platform should be CPU");
                assertFalse(platform.isGpu(), "CPU platform should not be GPU");
            } else {
                assertFalse(platform.isCpu(), "GPU platform should not be CPU");
                assertTrue(platform.isGpu(), "GPU platform should be GPU");
            }
        }
    }
}
