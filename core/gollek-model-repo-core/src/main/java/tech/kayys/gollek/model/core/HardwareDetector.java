/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */

package tech.kayys.gollek.model.core;

import org.eclipse.microprofile.config.inject.ConfigProperty;

/**
 * Hardware capability detector
 */
public class HardwareDetector {

    @ConfigProperty(name = "hardware.cuda.enabled", defaultValue = "false")
    boolean cudaEnabled;

    @ConfigProperty(name = "hardware.memory.available", defaultValue = "8589934592") // 8GB
    long availableMemory;

    public HardwareCapabilities detect() {
        return HardwareCapabilities.builder()
                .hasCUDA(cudaEnabled && isCUDAAvailable())
                .availableMemory(availableMemory)
                .cpuCores(Runtime.getRuntime().availableProcessors())
                .build();
    }

    private boolean isCUDAAvailable() {
        if (!cudaEnabled) {
            return false;
        }

        // 1. Check for NVIDIA Management Library (NVML) which is part of the driver
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("win") ? "nvml.dll" : "libnvidia-ml.so";

        // Try to find the library in common locations or system path
        // This is a heuristic but fairly reliable for driver presence
        boolean hasLib = false;
        if (os.contains("win")) {
            hasLib = java.nio.file.Files
                    .exists(java.nio.file.Paths.get(System.getenv("SystemRoot"), "System32", libName));
        } else {
            hasLib = java.nio.file.Files.exists(java.nio.file.Paths.get("/usr/lib/x86_64-linux-gnu", libName)) ||
                    java.nio.file.Files.exists(java.nio.file.Paths.get("/usr/lib64", libName));
        }

        if (hasLib) {
            return true;
        }

        // 2. Fallback: Try executing nvidia-smi
        try {
            Process process = new ProcessBuilder("nvidia-smi", "-L").start();
            return process.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

}