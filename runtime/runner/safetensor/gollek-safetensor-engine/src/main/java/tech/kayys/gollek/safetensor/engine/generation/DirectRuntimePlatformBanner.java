/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.inject.Instance;

/**
 * User-facing runtime platform banner for direct generation paths.
 */
final class DirectRuntimePlatformBanner {
    private DirectRuntimePlatformBanner() {
    }

    static void print(Instance<?> metalBackend) {
        Status status = describe(DirectInferenceProfiler.metalDeviceLabel(metalBackend));
        // System.out.println(status.platformLine());
        // System.out.println(status.accelerationLine());
        System.out.flush();
    }

    static Status describe(String metalDevice) {
        if (metalDevice != null && !metalDevice.contains("CPU")) {
            return new Status("Platform: Metal", "✓ GPU acceleration enabled (" + metalDevice + ")");
        }
        return new Status("Platform: Apple Silicon", "✓ CPU acceleration enabled (Accelerate AMX)");
    }

    record Status(String platformLine, String accelerationLine) {
    }
}
