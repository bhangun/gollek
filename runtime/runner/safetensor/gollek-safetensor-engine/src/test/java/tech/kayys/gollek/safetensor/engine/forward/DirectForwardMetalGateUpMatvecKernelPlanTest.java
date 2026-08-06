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

class DirectForwardMetalGateUpMatvecKernelPlanTest {

    @Test
    void describesSiluHalfKernel() {
        DirectForwardMetalGateUpMatvecKernelPlan plan = plan(FFNActivationType.SILU, false);

        assertEquals("swiglu_gate_up_matvec_f16", plan.pathSuffix());
        assertEquals("accept:swiglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesSiluBf16Kernel() {
        DirectForwardMetalGateUpMatvecKernelPlan plan = plan(FFNActivationType.SILU, true);

        assertEquals("swiglu_gate_up_matvec_bf16", plan.pathSuffix());
        assertEquals("accept:swiglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void describesGeluHalfKernel() {
        DirectForwardMetalGateUpMatvecKernelPlan plan = plan(FFNActivationType.GELU, false);

        assertEquals("geglu_gate_up_matvec_f16", plan.pathSuffix());
        assertEquals("accept:geglu:native_bf16=false", plan.acceptDecision());
    }

    @Test
    void describesGeluBf16Kernel() {
        DirectForwardMetalGateUpMatvecKernelPlan plan = plan(FFNActivationType.GELU, true);

        assertEquals("geglu_gate_up_matvec_bf16", plan.pathSuffix());
        assertEquals("accept:geglu:native_bf16=true", plan.acceptDecision());
    }

    @Test
    void rejectsUnsupportedActivation() {
        DirectForwardMetalFfnActivationPlan unsupported = DirectForwardMetalFfnActivationPlan.from(null);

        assertThrows(IllegalArgumentException.class,
                () -> DirectForwardMetalGateUpMatvecKernelPlan.from(unsupported, false));
    }

    private static DirectForwardMetalGateUpMatvecKernelPlan plan(
            FFNActivationType activationType,
            boolean nativeBf16Weights) {
        return DirectForwardMetalGateUpMatvecKernelPlan.from(
                DirectForwardMetalFfnActivationPlan.from(activationType),
                nativeBf16Weights);
    }
}
