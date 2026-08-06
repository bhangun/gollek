/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectRuntimePlatformBannerTest {

    @Test
    void describesMetalGpuDevice() {
        DirectRuntimePlatformBanner.Status status = DirectRuntimePlatformBanner.describe("Apple M4");

        assertEquals("Platform: Metal", status.platformLine());
        assertEquals("✓ GPU acceleration enabled (Apple M4)", status.accelerationLine());
    }

    @Test
    void fallsBackToCpuWhenNoMetalDeviceIsAvailable() {
        DirectRuntimePlatformBanner.Status status = DirectRuntimePlatformBanner.describe(null);

        assertEquals("Platform: Apple Silicon", status.platformLine());
        assertEquals("✓ CPU acceleration enabled (Accelerate AMX)", status.accelerationLine());
    }

    @Test
    void treatsCpuNamedDeviceAsCpuFallback() {
        DirectRuntimePlatformBanner.Status status = DirectRuntimePlatformBanner.describe("CPU");

        assertEquals("Platform: Apple Silicon", status.platformLine());
        assertEquals("✓ CPU acceleration enabled (Accelerate AMX)", status.accelerationLine());
    }
}
