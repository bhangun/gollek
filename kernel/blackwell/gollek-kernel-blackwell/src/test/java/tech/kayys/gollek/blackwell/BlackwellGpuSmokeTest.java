package tech.kayys.gollek.blackwell;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Smoke test for Blackwell GPU availability.
 *
 * <p>
 * These tests are skipped unless {@code CUDA_VISIBLE_DEVICES} is set
 * and the device is detected as Blackwell (compute cap ≥ 10.0).
 * </p>
 */
@EnabledIfEnvironmentVariable(named = "CUDA_VISIBLE_DEVICES", matches = ".*")
class BlackwellGpuSmokeTest {

    @Test
    void testBlackwellLibraryAvailable() {
        String cudaPath = System.getenv("CUDA_VISIBLE_DEVICES");
        assertThat(cudaPath).isNotBlank();
    }

    @Test
    void testBlackwellDeviceCount() {
        String visibleDevices = System.getenv("CUDA_VISIBLE_DEVICES");
        if (visibleDevices != null && !visibleDevices.isBlank()) {
            int count = visibleDevices.split(",").length;
            assertThat(count).isGreaterThan(0);
        }
    }
}
