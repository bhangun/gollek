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

class DirectForwardMetalMatvecGatedFfnKernelPlanTest {

    @Test
    void describesSiluHalfKernel() {
        DirectForwardMetalMatvecGatedFfnKernelPlan plan = plan(FFNActivationType.SILU, false);

        assertEquals("swiglu_matvec_f16", plan.pathSuffix());
        assertEquals("ffn_swiglu_matvec_metal", plan.profilerKey());
        assertEquals("accept:swiglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesSiluBf16Kernel() {
        DirectForwardMetalMatvecGatedFfnKernelPlan plan = plan(FFNActivationType.SILU, true);

        assertEquals("swiglu_matvec_bf16", plan.pathSuffix());
        assertEquals("ffn_swiglu_matvec_bf16", plan.profilerKey());
        assertEquals("accept:swiglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void describesGeluHalfKernel() {
        DirectForwardMetalMatvecGatedFfnKernelPlan plan = plan(FFNActivationType.GELU, false);

        assertEquals("geglu_matvec_f16", plan.pathSuffix());
        assertEquals("ffn_geglu_matvec_metal", plan.profilerKey());
        assertEquals("accept:geglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesGeluBf16Kernel() {
        DirectForwardMetalMatvecGatedFfnKernelPlan plan = plan(FFNActivationType.GELU, true);

        assertEquals("geglu_matvec_bf16", plan.pathSuffix());
        assertEquals("ffn_geglu_matvec_bf16", plan.profilerKey());
        assertEquals("accept:geglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void rejectsUnsupportedActivation() {
        DirectForwardMetalFfnActivationPlan unsupported = DirectForwardMetalFfnActivationPlan.from(null);

        assertThrows(IllegalArgumentException.class,
                () -> DirectForwardMetalMatvecGatedFfnKernelPlan.from(unsupported, false));
    }

    private static DirectForwardMetalMatvecGatedFfnKernelPlan plan(
            FFNActivationType activationType,
            boolean nativeBf16Weights) {
        return DirectForwardMetalMatvecGatedFfnKernelPlan.from(
                DirectForwardMetalFfnActivationPlan.from(activationType),
                nativeBf16Weights);
    }
}
