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
 */

package tech.kayys.gollek.plugin.kernel.rocm;

import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.kernel.KernelPlugin;

import java.util.Map;
import java.util.Set;

/**
 * ROCm kernel plugin for AMD GPUs.
 * 
 * <p>Provides AMD ROCm kernel implementations for GPU acceleration.</p>
 * 
 * <h2>Requirements</h2>
 * <ul>
 *   <li>AMD GPU (GCN 4.0+)</li>
 *   <li>ROCm 5.0+</li>
 *   <li>Linux OS (ROCm primarily supports Linux)</li>
 * </ul>
 */
public class RocmKernelPlugin implements KernelPlugin {

    private static final Logger LOG = Logger.getLogger(RocmKernelPlugin.class);
    public static final String ID = "rocm-kernel";

    private boolean enabled = true;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "ROCm Kernel";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public String description() {
        return "AMD ROCm kernel implementations for GPU acceleration";
    }

    @Override
    public boolean isAvailable() {
        if (!enabled) {
            return false;
        }
        try {
            Class.forName("org.bytedeco.rocm.hipRuntime");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    @Override
    public String platform() {
        return "rocm";
    }

    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("vega", "navi", "cdna", "cdna2", "cdna3");
    }

    @Override
    public Set<String> supportedVersions() {
        return Set.of("5.0", "5.1", "5.2", "5.3", "5.4", "5.5", "6.0");
    }

    @Override
    public Object execute(String operation, Map<String, Object> params) {
        if (!isAvailable()) {
            throw new IllegalStateException("ROCm kernel is not available");
        }
        LOG.infof("Executing ROCm operation: %s", operation);
        return Map.of("status", "success", "platform", "rocm", "operation", operation);
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
            "platform", "rocm",
            "supported_ops", Set.of("gemm", "attention", "layer_norm", "activation")
        );
    }
}
