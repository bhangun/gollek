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

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.model.config.HardwareConfig;
import tech.kayys.gollek.spi.spec.HardwareCapabilities;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Hardware capability detector
 */
@ApplicationScoped
@io.quarkus.arc.DefaultBean
public class HardwareDetector {

    private static final Logger LOG = Logger.getLogger(HardwareDetector.class);

    @Inject
    HardwareConfig hardwareConfig;

    public HardwareCapabilities detect() {
        return HardwareCapabilities.builder()
                .hasCUDA(isCUDAAvailable())
                .availableMemory(hardwareConfig.availableMemory())
                .cpuCores(Runtime.getRuntime().availableProcessors())
                .build();
    }

    public boolean cudaEnabled() {
        return isCUDAAvailable();
    }

    public boolean hasCUDA() {
        return isCUDAAvailable();
    }

    private boolean isCUDAAvailable() {
        if (!hardwareConfig.cudaEnabled()) {
            LOG.info("CUDA is disabled by configuration");
            return false;
        }

        // 1. Check for NVIDIA Management Library (NVML) which is part of the driver
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("win") ? "nvml.dll" : "libnvidia-ml.so";

        // Try to find the library in common locations or system path
        // This is a heuristic but fairly reliable for driver presence
        boolean hasLib = false;
        if (os.contains("win")) {
            hasLib = Files
                    .exists(Paths.get(System.getenv("SystemRoot"), "System32", libName));
        } else {
            hasLib = Files.exists(Paths.get("/usr/lib/x86_64-linux-gnu", libName)) ||
                    Files.exists(Paths.get("/usr/lib64", libName));
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
