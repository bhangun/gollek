/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

/**
 * ServiceLoader extension point for detachable unified multimodal runtimes.
 */
public interface UnifiedMultimodalRuntime {
    UnifiedRuntimeManifest manifest();

    default String runtimeId() {
        UnifiedRuntimeManifest manifest = manifest();
        return manifest == null ? "unknown-unified-runtime" : manifest.runtimeId();
    }

    default boolean supportsModelType(String modelType) {
        UnifiedRuntimeManifest manifest = manifest();
        return manifest != null && manifest.supportsModelType(modelType);
    }

    default UnifiedEmbeddingPlan plan(UnifiedMultimodalRequest request) {
        throw new UnsupportedOperationException(
                "Unified multimodal runtime " + runtimeId() + " does not implement embedding planning yet.");
    }
}
