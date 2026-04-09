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

package tech.kayys.gollek.sdk.vision;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;
import tech.kayys.gollek.sdk.vision.layers.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for vision module backend execution.
 *
 * @author Gollek Team
 */
class VisionBackendIntegrationTest {

    private VisionBackendProvider backend;

    @BeforeEach
    void setUp() {
        // Clear registry to start fresh
        VisionBackendRegistry.clear();
        VisionBackendRegistry.ensureInitialized();
        
        // Get the default backend
        backend = VisionBackendRegistry.getDefault();
        assertNotNull(backend, "Default backend should be available");
    }

    @Test
    void testBackendRegistration() {
        assertTrue(VisionBackendRegistry.isAvailable("cpu"), "CPU backend should be available");
        
        VisionBackendProvider cpuBackend = VisionBackendRegistry.get("cpu");
        assertNotNull(cpuBackend, "Should retrieve CPU backend");
        assertEquals("cpu", cpuBackend.getBackendId());
    }

    @Test
    void testDefaultBackendSelection() {
        VisionBackendProvider defaultBackend = VisionBackendRegistry.getDefault();
        assertNotNull(defaultBackend, "Should have default backend");
        assertTrue(defaultBackend.isAvailable(), "Default backend should be available");
    }

    @Test
    void testBackendDiagnostics() {
        String info = VisionBackendRegistry.getDiagnosticsInfo();
        assertNotNull(info);
        assertTrue(info.contains("Vision Backends"), "Should contain backend info");
        System.out.println(info);
    }

    @Test
    void testConv2dWithBackend() {
        // Create Conv2d layer
        Conv2d conv = new Conv2d(3, 16, 3, 1, 1);
        
        // Create input tensor
        GradTensor input = GradTensor.randn(1, 3, 32, 32);
        
        // Forward pass should dispatch to backend
        assertDoesNotThrow(() -> {
            GradTensor output = conv.forward(input);
            assertNotNull(output, "Output should not be null");
            assertEquals(4, output.shape().length, "Output should be 4D");
            assertEquals(16, output.shape()[1], "Output channels should match");
        }, "Conv2d forward should execute without error");
    }

    @Test
    void testConv2dOutputShape() {
        Conv2d conv = new Conv2d(3, 32, 3, 2, 1);
        GradTensor input = GradTensor.randn(4, 3, 64, 64);
        GradTensor output = conv.forward(input);
        
        // Check output shape
        long[] expectedShape = {4, 32, 32, 32};  // N, C, H/2, W/2 (stride=2)
        assertArrayEquals(expectedShape, output.shape(), "Output shape mismatch");
    }

    @Test
    void testBackendMemoryManagement() {
        VisionBackendProvider provider = VisionBackendRegistry.getDefault();
        
        // Clear memory
        provider.clearMemory();
        assertEquals(0, provider.getMemoryUsage(), "Memory should be cleared");
    }

    @Test
    void testBackendOptimizations() {
        VisionBackendProvider provider = VisionBackendRegistry.getDefault();
        
        // Test kernel fusion setting
        provider.setKernelFusionEnabled(true);
        assertTrue(provider.isKernelFusionEnabled(), "Kernel fusion should be enabled");
        
        provider.setKernelFusionEnabled(false);
        assertFalse(provider.isKernelFusionEnabled(), "Kernel fusion should be disabled");
    }

    @Test
    void testBackendMemoryStrategy() {
        VisionBackendProvider provider = VisionBackendRegistry.getDefault();
        
        // Test memory strategy
        provider.setMemoryStrategy("pool");
        assertEquals("pool", provider.getMemoryStrategy());
        
        provider.setMemoryStrategy("malloc");
        assertEquals("malloc", provider.getMemoryStrategy());
    }

    @Test
    void testMultipleConv2dLayers() {
        Conv2d conv1 = new Conv2d(3, 16, 3, 1, 1);
        Conv2d conv2 = new Conv2d(16, 32, 3, 1, 1);
        Conv2d conv3 = new Conv2d(32, 64, 3, 2, 1);
        
        GradTensor input = GradTensor.randn(2, 3, 64, 64);
        
        // Forward through network
        GradTensor x = conv1.forward(input);
        assertEquals(16, x.shape()[1]);
        
        x = conv2.forward(x);
        assertEquals(32, x.shape()[1]);
        
        x = conv3.forward(x);
        assertEquals(64, x.shape()[1]);
        assertEquals(32, x.shape()[2], "Height should be halved due to stride=2");
    }

    @Test
    void testBackendAvailability() {
        // CPU backend is always available
        assertTrue(VisionBackendRegistry.isAvailable("cpu"));
        
        // Non-existent backend
        assertFalse(VisionBackendRegistry.isAvailable("nonexistent"));
    }

    @Test
    void testBackendPriority() {
        VisionBackendProvider provider = VisionBackendRegistry.getDefault();
        assertNotNull(provider.getPriority());
    }

    @Test
    void testBatchNorm2dWithBackend() {
        // Create BatchNorm2d layer
        BatchNorm2d bn = new BatchNorm2d(32);
        
        // Create input tensor
        GradTensor input = GradTensor.randn(2, 32, 64, 64);
        
        // Forward pass should dispatch to backend
        assertDoesNotThrow(() -> {
            GradTensor output = bn.forward(input);
            assertNotNull(output, "Output should not be null");
            assertArrayEquals(input.shape(), output.shape(), "Output shape should match input shape");
        }, "BatchNorm2d forward should execute without error");
    }

    @Test
    void testMaxPool2dWithBackend() {
        // Create MaxPool2d layer
        MaxPool2d pool = new MaxPool2d(2, 2, 0);
        
        // Create input tensor
        GradTensor input = GradTensor.randn(1, 16, 64, 64);
        
        // Forward pass should dispatch to backend
        assertDoesNotThrow(() -> {
            GradTensor output = pool.forward(input);
            assertNotNull(output, "Output should not be null");
            assertEquals(4, output.shape().length, "Output should be 4D");
            assertEquals(32, output.shape()[2], "Height should be halved");
            assertEquals(32, output.shape()[3], "Width should be halved");
        }, "MaxPool2d forward should execute without error");
    }

    @Test
    void testAdaptiveAvgPool2dWithBackend() {
        // Create AdaptiveAvgPool2d layer (global average pooling)
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(1);
        
        // Create input tensor
        GradTensor input = GradTensor.randn(2, 512, 7, 7);
        
        // Forward pass should dispatch to backend
        assertDoesNotThrow(() -> {
            GradTensor output = pool.forward(input);
            assertNotNull(output, "Output should not be null");
            long[] expectedShape = {2, 512, 1, 1};
            assertArrayEquals(expectedShape, output.shape(), "Output shape mismatch");
        }, "AdaptiveAvgPool2d forward should execute without error");
    }

    @Test
    void testLinearWithBackend() {
        // Create Linear layer
        Linear linear = new Linear(512, 1000);
        
        // Create input tensor
        GradTensor input = GradTensor.randn(4, 512);
        
        // Forward pass should dispatch to backend
        assertDoesNotThrow(() -> {
            GradTensor output = linear.forward(input);
            assertNotNull(output, "Output should not be null");
            assertEquals(2, output.shape().length, "Output should be 2D");
            assertEquals(4, output.shape()[0], "Batch size should match");
            assertEquals(1000, output.shape()[1], "Output features should be 1000");
        }, "Linear forward should execute without error");
    }

    @Test
    void testFullNetworkWithBackend() {
        // Build a simple CNN with multiple layers using backend dispatch
        Conv2d conv1 = new Conv2d(3, 16, 3, 1, 1);
        BatchNorm2d bn1 = new BatchNorm2d(16);
        MaxPool2d pool1 = new MaxPool2d(2, 2);
        
        Conv2d conv2 = new Conv2d(16, 32, 3, 1, 1);
        BatchNorm2d bn2 = new BatchNorm2d(32);
        MaxPool2d pool2 = new MaxPool2d(2, 2);
        
        AdaptiveAvgPool2d globalPool = new AdaptiveAvgPool2d(1);
        Linear classifier = new Linear(32, 10);
        
        // Create input
        GradTensor input = GradTensor.randn(2, 3, 64, 64);
        
        // Forward through network - all layers should dispatch to backend
        GradTensor x = conv1.forward(input);
        assertEquals(16, x.shape()[1]);
        
        x = bn1.forward(x);
        assertEquals(16, x.shape()[1]);
        
        x = pool1.forward(x);
        assertEquals(32, x.shape()[2], "Height should be 32");
        
        x = conv2.forward(x);
        assertEquals(32, x.shape()[1]);
        
        x = bn2.forward(x);
        assertEquals(32, x.shape()[1]);
        
        x = pool2.forward(x);
        assertEquals(16, x.shape()[2], "Height should be 16");
        
        x = globalPool.forward(x);
        assertEquals(1, x.shape()[2], "Height should be 1");
        
        // Reshape for classifier: [N, C, 1, 1] -> [N, C]
        x = classifier.forward(x.reshape(x.shape()[0], x.shape()[1]));
        assertEquals(10, x.shape()[1]);
    }
}
