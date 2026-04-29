package tech.kayys.gollek.spi.tensor;

import tech.kayys.gollek.spi.model.DeviceType;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ComputeKernelRegistry and CpuKernel.
 */
class ComputeKernelRegistryTest {

    @BeforeEach
    @AfterEach
    void setUp() {
        // Clear registry before/after each test
        ComputeKernelRegistry.get().clear();
    }

    @Test
    @DisplayName("Registry returns singleton instance")
    void testSingletonInstance() {
        ComputeKernelRegistry registry1 = ComputeKernelRegistry.get();
        ComputeKernelRegistry registry2 = ComputeKernelRegistry.get();
        
        assertSame(registry1, registry2);
    }

    @Test
    @DisplayName("CPU kernel is always available")
    void testCpuKernelAlwaysAvailable() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        assertNotNull(kernel);
        assertEquals(DeviceType.CPU, kernel.deviceType());
        assertTrue(kernel.isAvailable());
    }

    @Test
    @DisplayName("CPU kernel device name is descriptive")
    void testCpuKernelDeviceName() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        String deviceName = kernel.deviceName();
        
        assertNotNull(deviceName);
        assertTrue(deviceName.contains("CPU"));
        assertTrue(deviceName.contains("cores"));
    }

    @Test
    @DisplayName("CPU kernel memory reporting works")
    void testCpuKernelMemory() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        assertTrue(kernel.totalMemory() > 0);
        assertTrue(kernel.availableMemory() > 0);
        assertTrue(kernel.availableMemory() <= kernel.totalMemory());
    }

    @Test
    @DisplayName("CPU kernel allocates and frees memory")
    void testCpuKernelAllocate() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        MemorySegment segment = kernel.allocate(1024);
        assertNotNull(segment);
        assertEquals(1024, segment.byteSize());
        
        // Free should not throw
        assertDoesNotThrow(() -> kernel.free(segment));
    }

    @Test
    @DisplayName("CPU kernel matmul produces correct results")
    void testCpuKernelMatmul() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        // 2x2 matrices
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] expected = {19.0f, 22.0f, 43.0f, 50.0f};
        
        MemorySegment aSeg = MemorySegment.ofArray(a);
        MemorySegment bSeg = MemorySegment.ofArray(b);
        MemorySegment cSeg = kernel.allocate(4 * Float.BYTES);
        
        kernel.matmul(cSeg, aSeg, bSeg, 2, 2, 2);
        
        float[] result = new float[4];
        for (int i = 0; i < 4; i++) {
            result[i] = cSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        
        for (int i = 0; i < 4; i++) {
            assertEquals(expected[i], result[i], 0.001f, 
                "Mismatch at index " + i);
        }
        
        kernel.free(cSeg);
    }

    @Test
    @DisplayName("CPU kernel elementwise operations work")
    void testCpuKernelElementwise() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] c = new float[4];
        
        MemorySegment aSeg = MemorySegment.ofArray(a);
        MemorySegment bSeg = MemorySegment.ofArray(b);
        MemorySegment cSeg = MemorySegment.ofArray(c);
        
        // Test add
        kernel.elementwiseAdd(cSeg, aSeg, bSeg, 4);
        assertEquals(6.0f, c[0], 0.001f);
        assertEquals(8.0f, c[1], 0.001f);
        
        // Test mul
        kernel.elementwiseMul(cSeg, aSeg, bSeg, 4);
        assertEquals(5.0f, c[0], 0.001f);
        assertEquals(12.0f, c[1], 0.001f);
        
        // Test scale
        kernel.elementwiseScale(aSeg, 2.0f, 4);
        assertEquals(2.0f, a[0], 0.001f);
        assertEquals(4.0f, a[1], 0.001f);
    }

    @Test
    @DisplayName("CPU kernel SiLU activation works")
    void testCpuKernelSilu() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        float[] input = {0.0f, 1.0f, -1.0f};
        float[] output = new float[3];
        
        MemorySegment inputSeg = MemorySegment.ofArray(input);
        MemorySegment outputSeg = MemorySegment.ofArray(output);
        
        kernel.silu(outputSeg, inputSeg, 3);
        
        // SiLU(0) = 0
        assertEquals(0.0f, output[0], 0.001f);
        
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        float expected = 1.0f / (1.0f + (float) Math.exp(-1.0));
        assertEquals(expected, output[1], 0.001f);
    }

    @Test
    @DisplayName("CPU kernel RMS Norm works")
    void testCpuKernelRmsNorm() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        float[] input = {3.0f, 4.0f};
        float[] weight = {1.0f, 1.0f};
        float[] output = new float[2];
        
        MemorySegment inputSeg = MemorySegment.ofArray(input);
        MemorySegment weightSeg = MemorySegment.ofArray(weight);
        MemorySegment outputSeg = MemorySegment.ofArray(output);
        
        kernel.rmsNorm(outputSeg, inputSeg, weightSeg, 2, 1e-5f, false);
        
        // RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.536
        // output[0] = 3 / 3.536 ≈ 0.848
        // output[1] = 4 / 3.536 ≈ 1.131
        assertEquals(0.848f, output[0], 0.01f);
        assertEquals(1.131f, output[1], 0.01f);
    }

    @Test
    @DisplayName("Copy operations work correctly")
    void testCpuKernelCopyOperations() {
        ComputeKernel kernel = ComputeKernelRegistry.get().getBestAvailable();
        
        byte[] src = {1, 2, 3, 4, 5};
        byte[] dst = new byte[5];
        
        MemorySegment srcSeg = MemorySegment.ofArray(src);
        MemorySegment dstSeg = MemorySegment.ofArray(dst);
        
        kernel.copyHostToDevice(dstSeg, srcSeg, 5);
        assertArrayEquals(src, dst);
        
        byte[] dst2 = new byte[5];
        MemorySegment dst2Seg = MemorySegment.ofArray(dst2);
        kernel.copyDeviceToHost(dst2Seg, dstSeg, 5);
        assertArrayEquals(src, dst2);
    }

    @Test
    @DisplayName("Register custom kernel supplier")
    void testRegisterCustomKernel() {
        ComputeKernelRegistry registry = ComputeKernelRegistry.get();
        
        registry.register(DeviceType.CUDA, () -> {
            throw new IllegalStateException("CUDA not available");
        });
        
        // Should not affect CPU availability
        ComputeKernel kernel = registry.getBestAvailable();
        assertNotNull(kernel);
    }
}
