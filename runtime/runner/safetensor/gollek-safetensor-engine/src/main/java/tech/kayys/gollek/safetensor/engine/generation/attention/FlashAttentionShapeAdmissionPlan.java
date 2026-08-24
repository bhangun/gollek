/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Arrays;

record FlashAttentionShapeAdmissionPlan(
        Integer resolvedHeadDim,
        String rejectionMessage) {

    static FlashAttentionShapeAdmissionPlan separateQueryWeightLayout(
            AccelTensor queryWeight,
            ModelConfig config,
            int numQueryHeads,
            int layerIdx) {
        if (queryWeight == null || numQueryHeads <= 0) {
            return admit(config == null ? 0 : config.resolvedHeadDim());
        }
        long rows = queryWeight.size(0);
        if (rows % numQueryHeads == 0) {
            return admit(Math.toIntExact(rows / numQueryHeads));
        }
        return reject("Attention query projection layout mismatch at layer " + layerIdx
                + ": rows=" + rows
                + " numQueryHeads=" + numQueryHeads
                + " configuredHeadDim=" + resolvedHeadDim(config)
                + " hiddenSize=" + hiddenSize(config)
                + " modelType=" + modelType(config)
                + " architectures=" + architectures(config)
                + " weightShape=" + Arrays.toString(queryWeight.shape())
                + ". If this model uses fused/packed QKV, keep that policy in its model-family"
                + " runtime traits instead of using separate q/k/v projection layout.");
    }

    static FlashAttentionShapeAdmissionPlan packedQkvWeight(
            AccelTensor packedWeight,
            FlashAttentionHeadLayout layout) {
        if (packedWeight == null || layout == null || !layout.packedQkvProjection()) {
            return admit(null);
        }
        long rows = packedWeight.size(0);
        long expected = layout.packedQkvProjectionDim();
        if (rows == expected) {
            return admit(null);
        }
        return reject("Packed QKV projection rows do not match attention layout: rows=" + rows
                + " expected=" + expected
                + " qHeads=" + layout.numQueryHeads()
                + " kvHeads=" + layout.numKeyValueHeads()
                + " headDim=" + layout.headDim());
    }

    static FlashAttentionShapeAdmissionPlan packedQkvOutput(
            AccelTensor packedProjection,
            FlashAttentionHeadLayout layout) {
        if (layout == null || !layout.packedQkvProjection()) {
            return admit(null);
        }
        long expected = layout.packedQkvProjectionDim();
        if (packedProjection != null && packedProjection.size(-1) == expected) {
            return admit(null);
        }
        return reject("Packed QKV projection output does not match attention layout: lastDim="
                + (packedProjection == null ? "<missing>" : Long.toString(packedProjection.size(-1)))
                + " expected=" + expected);
    }

    boolean admitted() {
        return rejectionMessage == null;
    }

    IllegalArgumentException asException() {
        return new IllegalArgumentException(rejectionMessage);
    }

    private static FlashAttentionShapeAdmissionPlan admit(Integer resolvedHeadDim) {
        return new FlashAttentionShapeAdmissionPlan(resolvedHeadDim, null);
    }

    private static FlashAttentionShapeAdmissionPlan reject(String message) {
        return new FlashAttentionShapeAdmissionPlan(null, message);
    }

    private static int resolvedHeadDim(ModelConfig config) {
        return config == null ? 0 : config.resolvedHeadDim();
    }

    private static int hiddenSize(ModelConfig config) {
        return config == null ? 0 : config.hiddenSize();
    }

    private static String modelType(ModelConfig config) {
        return config == null ? "<unknown>" : config.modelType();
    }

    private static Object architectures(ModelConfig config) {
        return config == null ? null : config.architectures();
    }
}
