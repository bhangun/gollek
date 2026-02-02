package tech.kayys.golek.provider.litert;

import org.junit.jupiter.api.*;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.provider.core.spi.RunnerConfiguration;

import java.nio.file.Paths;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive integration test for all enhanced LiteRT features.
 */
@DisplayName("LiteRT Enhanced Features Integration Test")
@Timeout(30)
class LiteRTEnhancedFeaturesTest {

    private LiteRTCpuRunner runner;
    private ModelManifest manifest;
    private RunnerConfiguration config;

    @BeforeEach
    void setUp() {
        // Create a test model manifest
        manifest = ModelManifest.builder()
            .name("test-model")
            .version("1.0")
            .framework("litert")
            .storageUri("file://" + System.getProperty("user.dir") + "/src/test/resources/test-model.tflite")
            .build();

        // Create configuration with all features enabled
        config = RunnerConfiguration.builder()
            .parameter("useGpu", false)
            .parameter("useNpu", false)
            .parameter("useBatching", true)
            .parameter("useMemoryPooling", true)
            .parameter("useErrorHandling", true)
            .parameter("useMonitoring", true)
            .parameter("numThreads", 2)
            .build();
    }

    @AfterEach
    void tearDown() {
        if (runner != null) {
            runner.close();
        }
    }

    @Test
    @DisplayName("Runner should initialize with all enhanced features")
    void testInitializationWithAllFeatures() {
        assertDoesNotThrow(() -> {
            runner = new LiteRTCpuRunner();
            runner.initialize(manifest, config);
        });

        assertNotNull(runner);
        assertTrue(runner.health());
    }

    @Test
    @DisplayName("Capabilities should reflect all enhanced features")
    void testEnhancedCapabilities() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        var capabilities = runner.capabilities();
        
        assertTrue(capabilities.isSupportsBatching());
        assertTrue(capabilities.isSupportsQuantization());
        assertTrue(capabilities.isSupportsGpuAcceleration());
        assertTrue(capabilities.isSupportsNpuAcceleration());
        assertTrue(capabilities.isSupportsMemoryPooling());
        assertTrue(capabilities.isSupportsAdaptiveBatching());
        assertTrue(capabilities.isSupportsCircuitBreaker());
        assertTrue(capabilities.isSupportsAutomaticRetry());
        assertTrue(capabilities.isSupportsComprehensiveMonitoring());
        assertTrue(capabilities.isSupportsHealthChecks());
        assertTrue(capabilities.isSupportsPerformanceMetrics());
        
        assertEquals(128, capabilities.getMaxBatchSize());
        assertEquals(3, capabilities.getMaxRetries());
    }

    @Test
    @DisplayName("Delegate manager should be functional")
    void testDelegateManager() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Access the delegate manager through reflection (since it's private)
        assertDoesNotThrow(() -> {
            var field = runner.getClass().getDeclaredField("delegateManager");
            field.setAccessible(true);
            var delegateManager = (LiteRTDelegateManager) field.get(runner);
            
            assertNotNull(delegateManager);
            assertEquals(0, delegateManager.getDelegateCount());
            assertFalse(delegateManager.getDelegateInfo().isEmpty());
        });
    }

    @Test
    @DisplayName("Tensor utilities should work correctly")
    void testTensorUtilities() {
        // Test tensor validation
        assertTrue(LiteRTTensorUtils.validateShapeCompatibility(
            new long[]{1, 224, 224, 3}, 
            new long[]{1, 224, 224, 3}));

        assertFalse(LiteRTTensorUtils.validateShapeCompatibility(
            new long[]{1, 224, 224}, 
            new long[]{1, 224, 224, 3}));

        // Test element count calculation
        assertEquals(150528, LiteRTTensorUtils.calculateElementCount(new long[]{1, 224, 224, 3}));

        // Test byte size calculation
        assertEquals(150528 * 4, LiteRTTensorUtils.calculateByteSize(
            LiteRTNativeBindings.TfLiteType.FLOAT32, 
            new long[]{1, 224, 224, 3}));

        // Test data validation
        float[] floatData = {1.0f, 2.0f, 3.0f};
        byte[] floatBytes = LiteRTTensorUtils.floatArrayToBytes(floatData);
        assertTrue(LiteRTTensorUtils.validateTensorData(floatBytes, 
            LiteRTNativeBindings.TfLiteType.FLOAT32, floatBytes.length));
    }

    @Test
    @DisplayName("Batching manager should handle requests")
    void testBatchingManager() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Access the batching manager through reflection
        assertDoesNotThrow(() -> {
            var field = runner.getClass().getDeclaredField("batchingManager");
            field.setAccessible(true);
            var batchingManager = (LiteRTBatchingManager) field.get(runner);
            
            assertNotNull(batchingManager);
            assertTrue(batchingManager.isHealthy());
            assertEquals(0, batchingManager.getQueueSize());
            assertEquals(0, batchingManager.getPendingCount());
        });
    }

    @Test
    @DisplayName("Memory pool should manage allocations")
    void testMemoryPool() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Access the memory pool through reflection
        assertDoesNotThrow(() -> {
            var field = runner.getClass().getDeclaredField("memoryPool");
            field.setAccessible(true);
            var memoryPool = (LiteRTMemoryPool) field.get(runner);
            
            assertNotNull(memoryPool);
            assertTrue(memoryPool.isHealthy());
            
            // Test allocation and release
            var segment = memoryPool.allocate(1024);
            assertNotNull(segment);
            assertEquals(1024, segment.byteSize());
            
            memoryPool.release(segment);
            
            var stats = memoryPool.getStatistics();
            assertTrue(stats.getTotalAllocations() > 0);
            assertTrue(stats.getTotalReuses() >= 0);
        });
    }

    @Test
    @DisplayName("Error handler should classify errors correctly")
    void testErrorHandler() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Access the error handler through reflection
        assertDoesNotThrow(() -> {
            var field = runner.getClass().getDeclaredField("errorHandler");
            field.setAccessible(true);
            var errorHandler = (LiteRTErrorHandler) field.get(runner);
            
            assertNotNull(errorHandler);
            
            // Test error handling
            var strategy = errorHandler.handleError(
                new RuntimeException("Test error"), 
                "test_operation"
            );
            
            assertNotNull(strategy);
            assertFalse(strategy.shouldRetry()); // Unknown errors should not retry
            
            var stats = errorHandler.getStatistics();
            assertEquals(1, stats.getTotalErrors());
            assertEquals(0, stats.getRecoverableErrors());
            assertEquals(1, stats.getUnrecoverableErrors());
        });
    }

    @Test
    @DisplayName("Monitoring system should track metrics")
    void testMonitoringSystem() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Access the monitoring system through reflection
        assertDoesNotThrow(() -> {
            var field = runner.getClass().getDeclaredField("monitoring");
            field.setAccessible(true);
            var monitoring = (LiteRTMonitoring) field.get(runner);
            
            assertNotNull(monitoring);
            
            // Test recording metrics
            monitoring.recordSuccess("test_operation", 100, 50);
            monitoring.recordMemoryAllocation(1024);
            
            var stats = monitoring.getStatistics();
            assertEquals(1, stats.getTotalRequests());
            assertEquals(1, stats.getSuccessfulRequests());
            assertEquals(0, stats.getFailedRequests());
            assertEquals(100.0, stats.getAverageLatency(), 0.01);
            assertEquals(1024, stats.getCurrentMemoryUsage());
            
            // Test health check
            var healthStatus = monitoring.performHealthCheck();
            assertEquals(LiteRTMonitoring.HealthStatus.HEALTHY, healthStatus);
        });
    }

    @Test
    @DisplayName("Inference should work with enhanced features")
    void testInferenceWithEnhancedFeatures() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        // Create a simple test request
        var request = InferenceRequest.builder()
            .requestId("test-request-1")
            .modelId("test-model:1.0")
            .build();

        // Test inference execution
        assertDoesNotThrow(() -> {
            var response = runner.infer(request);
            assertNotNull(response);
            assertEquals("test-request-1", response.getRequestId());
            assertTrue(response.getLatencyMs() >= 0);
        });
    }

    @Test
    @DisplayName("Async inference should work correctly")
    void testAsyncInference() throws Exception {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        var request = InferenceRequest.builder()
            .requestId("test-async-request")
            .modelId("test-model:1.0")
            .build();

        CompletableFuture<InferenceResponse> future = runner.inferAsync(request);
        
        assertNotNull(future);
        assertFalse(future.isDone());
        
        // Wait for completion
        InferenceResponse response = future.get(5, TimeUnit.SECONDS);
        
        assertNotNull(response);
        assertEquals("test-async-request", response.getRequestId());
        assertTrue(response.getLatencyMs() >= 0);
    }

    @Test
    @DisplayName("Health checks should work correctly")
    void testHealthChecks() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        assertTrue(runner.health());
        assertNotNull(runner.healthStatus());
        assertTrue(runner.healthStatus().isHealthy());
    }

    @Test
    @DisplayName("Metrics should be available")
    void testMetrics() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        var metrics = runner.metrics();
        
        assertNotNull(metrics);
        assertTrue(metrics.getTotalRequests() >= 0);
        assertTrue(metrics.getFailedRequests() >= 0);
        assertTrue(metrics.getAverageLatencyMs() >= 0);
    }

    @Test
    @DisplayName("Runner should handle missing model gracefully")
    void testMissingModelHandling() {
        var invalidManifest = ModelManifest.builder()
            .name("missing-model")
            .version("1.0")
            .framework("litert")
            .storageUri("file:///nonexistent/model.tflite")
            .build();

        var exception = assertThrows(Exception.class, () -> {
            runner = new LiteRTCpuRunner();
            runner.initialize(invalidManifest, config);
        });

        assertTrue(exception.getMessage().contains("Model file not found"));
    }

    @Test
    @DisplayName("Configuration should be flexible")
    void testConfigurationFlexibility() {
        // Test with minimal configuration
        var minimalConfig = RunnerConfiguration.builder()
            .parameter("numThreads", 1)
            .build();

        assertDoesNotThrow(() -> {
            runner = new LiteRTCpuRunner();
            runner.initialize(manifest, minimalConfig);
        });

        assertTrue(runner.health());
    }

    @Test
    @DisplayName("Cleanup should work without errors")
    void testCleanup() {
        runner = new LiteRTCpuRunner();
        runner.initialize(manifest, config);

        assertDoesNotThrow(() -> {
            runner.close();
        });

        // Verify runner is no longer healthy after cleanup
        assertFalse(runner.health());
    }

    /**
     * Helper method to find test library.
     */
    private String findTestLibrary() {
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("linux") ? "libtensorflowlite_c.so"
                : os.contains("mac") ? "libtensorflowlite_c.dylib" 
                : "tensorflowlite_c.dll";

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

        return "/usr/local/lib/" + libName; // Default fallback
    }
}