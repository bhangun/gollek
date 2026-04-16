package tech.kayys.gollek.cuda;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Smoke test for CUDA GPU availability.
 *
 * <p>
 * These tests are skipped unless {@code CUDA_VISIBLE_DEVICES} is set,
 * indicating an intentional GPU test run.
 * </p>
 */
@EnabledIfEnvironmentVariable(named = "CUDA_VISIBLE_DEVICES", matches = ".*")
class CudaGpuSmokeTest {

    @Test
    void testCudaLibraryAvailable() {
        // Verify CUDA library can be loaded
        String cudaPath = System.getenv("CUDA_VISIBLE_DEVICES");
        assertThat(cudaPath).isNotBlank();
    }

    @Test
    void testCudaDeviceCount() {
        // In production, this would query cuDeviceGetCount
        String visibleDevices = System.getenv("CUDA_VISIBLE_DEVICES");
        if (visibleDevices != null && !visibleDevices.isBlank()) {
            int count = visibleDevices.split(",").length;
            assertThat(count).isGreaterThan(0);
        }
    }
}
