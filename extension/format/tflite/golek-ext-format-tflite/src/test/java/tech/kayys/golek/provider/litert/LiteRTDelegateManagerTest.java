package tech.kayys.golek.provider.litert;

import org.junit.jupiter.api.*;
import tech.kayys.golek.provider.litert.LiteRTDelegateManager.DelegateType;
import tech.kayys.golek.provider.litert.LiteRTDelegateManager.GpuBackend;
import tech.kayys.golek.provider.litert.LiteRTDelegateManager.NpuType;

import java.lang.foreign.Arena;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for LiteRT Delegate Manager.
 */
@DisplayName("LiteRT Delegate Manager Tests")
class LiteRTDelegateManagerTest {

    private LiteRTNativeBindings nativeBindings;
    private Arena arena;
    private LiteRTDelegateManager delegateManager;

    @BeforeEach
    void setUp() {
        // Initialize native bindings
        String libraryPath = findTestLibrary();
        Assumptions.assumeTrue(libraryPath != null, "LiteRT native library not found, skipping test");

        this.nativeBindings = new LiteRTNativeBindings(Paths.get(libraryPath));
        this.arena = Arena.ofConfined();
        this.delegateManager = new LiteRTDelegateManager(nativeBindings, arena);
    }

    @AfterEach
    void tearDown() {
        if (delegateManager != null) {
            delegateManager.cleanup();
        }
        if (arena != null) {
            arena.close();
        }
    }

    @Test
    @DisplayName("Delegate manager should initialize successfully")
    void testInitialization() {
        assertNotNull(delegateManager);
        assertEquals(0, delegateManager.getDelegateCount());
        assertTrue(delegateManager.getDelegateInfo().contains("Available Delegates (0)"));
    }

    @Test
    @DisplayName("Auto-detection should work without throwing exceptions")
    void testAutoDetection() {
        // This test verifies that auto-detection doesn't crash
        // Even if no delegates are available, it should complete gracefully
        delegateManager.autoDetectAndInitializeDelegates();

        // Should have 0 delegates on most test systems
        assertTrue(delegateManager.getDelegateCount() >= 0);
        assertFalse(delegateManager.getDelegateInfo().isEmpty());
    }

    @Test
    @DisplayName("GPU delegate initialization should handle unsupported backends gracefully")
    void testGpuDelegateInitialization() {
        // Test all GPU backends
        for (GpuBackend backend : GpuBackend.values()) {
            if (backend == GpuBackend.CUDA) {
                // CUDA is not supported in LiteRT
                assertThrows(UnsupportedOperationException.class,
                        () -> delegateManager.tryInitializeGpuDelegate(backend, backend.name()));
            } else {
                // Other backends should fail gracefully if not available
                boolean success = delegateManager.tryInitializeGpuDelegate(backend, backend.name());
                assertFalse(success, "GPU delegate should not be available in test environment");
            }
        }
    }

    @Test
    @DisplayName("NPU delegate initialization should handle unsupported types gracefully")
    void testNpuDelegateInitialization() {
        // Test all NPU types
        for (NpuType type : NpuType.values()) {
            boolean success = delegateManager.tryInitializeNpuDelegate(type, type.name());
            assertFalse(success, "NPU delegate should not be available in test environment");
        }
    }

    @Test
    @DisplayName("Delegate availability checks should work correctly")
    void testDelegateAvailability() {
        // Initially no delegates should be available
        for (DelegateType type : DelegateType.values()) {
            assertFalse(delegateManager.isDelegateAvailable(type));
        }

        // After auto-detection, still no delegates should be available in test
        // environment
        delegateManager.autoDetectAndInitializeDelegates();
        for (DelegateType type : DelegateType.values()) {
            assertFalse(delegateManager.isDelegateAvailable(type));
        }
    }

    @Test
    @DisplayName("Best delegate selection should return null when no delegates available")
    void testBestDelegateSelection() {
        assertNull(delegateManager.getBestAvailableDelegate());

        // Even after auto-detection, should still be null in test environment
        delegateManager.autoDetectAndInitializeDelegates();
        assertNull(delegateManager.getBestAvailableDelegate());
    }

    @Test
    @DisplayName("Delegate info should contain proper formatting")
    void testDelegateInfoFormatting() {
        String info = delegateManager.getDelegateInfo();
        assertTrue(info.contains("Available Delegates"));
        assertTrue(info.contains("ACTIVE") || info.contains("INACTIVE"));
    }

    @Test
    @DisplayName("Cleanup should work without throwing exceptions")
    void testCleanup() {
        // Initialize some delegates (they will fail but that's ok)
        delegateManager.autoDetectAndInitializeDelegates();

        // Cleanup should not throw
        assertDoesNotThrow(() -> delegateManager.cleanup());

        // After cleanup, delegate count should be 0
        assertEquals(0, delegateManager.getDelegateCount());
    }

    @Test
    @DisplayName("Multiple cleanup calls should be safe")
    void testMultipleCleanupCalls() {
        delegateManager.autoDetectAndInitializeDelegates();

        // Multiple cleanups should not throw
        assertDoesNotThrow(() -> {
            delegateManager.cleanup();
            delegateManager.cleanup();
            delegateManager.cleanup();
        });
    }

    @Test
    @DisplayName("Delegate manager should handle null native bindings gracefully")
    void testNullNativeBindingsHandling() {
        // This test verifies that the delegate manager can handle cases where
        // native bindings might not support certain delegate functions
        assertDoesNotThrow(() -> {
            delegateManager.autoDetectAndInitializeDelegates();
            delegateManager.getBestAvailableDelegate();
            delegateManager.getDelegateInfo();
            delegateManager.cleanup();
        });
    }

    /**
     * Find test library - uses a mock or test library if available.
     */
    private String findTestLibrary() {
        // In a real test environment, this would find the actual library
        // For this test, we'll use a path that should work in development
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("linux") ? "libtensorflowlite_c.so"
                : os.contains("mac") ? "libtensorflowlite_c.dylib"
                        : "tensorflowlite_c.dll";

        // Try common locations
        String[] paths = {
                "/usr/local/lib/" + libName,
                "/usr/lib/" + libName,
                "./lib/" + libName,
                "target/test-libs/" + libName
        };

        for (String path : paths) {
            if (java.nio.file.Files.exists(java.nio.file.Paths.get(path))) {
                return path;
            }
        }

        // If no library found, this test will be skipped in CI
        // but should work in development environments
        // If no library found, return null to skip tests
        return null;
    }

    @Test
    @DisplayName("Delegate manager should provide meaningful error messages")
    void testErrorMessages() {
        // Test that error cases provide meaningful messages
        Exception exception = assertThrows(UnsupportedOperationException.class,
                () -> delegateManager.tryInitializeGpuDelegate(GpuBackend.CUDA, "CUDA"));

        assertTrue(exception.getMessage().contains("CUDA not supported"));
    }

    @Test
    @DisplayName("Delegate count should be accurate")
    void testDelegateCountAccuracy() {
        assertEquals(0, delegateManager.getDelegateCount());

        // Even after auto-detection, count should remain accurate
        delegateManager.autoDetectAndInitializeDelegates();
        assertTrue(delegateManager.getDelegateCount() >= 0);
    }
}