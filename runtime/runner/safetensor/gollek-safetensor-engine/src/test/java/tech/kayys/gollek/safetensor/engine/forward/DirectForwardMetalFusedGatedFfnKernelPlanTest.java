/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

class DirectForwardMetalFusedGatedFfnKernelPlanTest {

    @Test
    void describesSiluHalfKernel() {
        DirectForwardMetalFusedGatedFfnKernelPlan plan = plan(FFNActivationType.SILU, false);

        assertEquals("swiglu_fused_f16", plan.pathSuffix());
        assertEquals("ffn_fused_metal", plan.profilerKey());
        assertEquals("accept:swiglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesSiluBf16Kernel() {
        DirectForwardMetalFusedGatedFfnKernelPlan plan = plan(FFNActivationType.SILU, true);

        assertEquals("swiglu_fused_bf16", plan.pathSuffix());
        assertEquals("ffn_fused_metal", plan.profilerKey());
        assertEquals("accept:swiglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void describesGeluHalfKernel() {
        DirectForwardMetalFusedGatedFfnKernelPlan plan = plan(FFNActivationType.GELU, false);

        assertEquals("geglu_fused_f16", plan.pathSuffix());
        assertEquals("ffn_geglu_fused_metal", plan.profilerKey());
        assertEquals("accept:geglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesGeluBf16Kernel() {
        DirectForwardMetalFusedGatedFfnKernelPlan plan = plan(FFNActivationType.GELU, true);

        assertEquals("geglu_fused_bf16", plan.pathSuffix());
        assertEquals("ffn_geglu_fused_metal", plan.profilerKey());
        assertEquals("accept:geglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void rejectsUnsupportedActivation() {
        DirectForwardMetalFfnActivationPlan unsupported = DirectForwardMetalFfnActivationPlan.from(null);

        assertThrows(IllegalArgumentException.class,
                () -> DirectForwardMetalFusedGatedFfnKernelPlan.from(unsupported, false));
    }

    private static DirectForwardMetalFusedGatedFfnKernelPlan plan(
            FFNActivationType activationType,
            boolean nativeBf16Weights) {
        return DirectForwardMetalFusedGatedFfnKernelPlan.from(
                DirectForwardMetalFfnActivationPlan.from(activationType),
                nativeBf16Weights);
    }
}
