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

class DirectForwardMetalMatvecRowsGatedFfnKernelPlanTest {

    @Test
    void describesSiluRowsBf16Kernel() {
        DirectForwardMetalMatvecRowsGatedFfnKernelPlan plan = plan(FFNActivationType.SILU);

        assertEquals("swiglu_matvec_rows_bf16_x4", plan.pathSuffix(2));
        assertEquals("ffn_swiglu_matvec_rows_bf16_x4", plan.profilerKey(2));
        assertEquals("accept:swiglu:native_bf16=true:native_rows=12:variant=x4",
                plan.acceptDecision(12, 2));
    }

    @Test
    void describesGeluRowsBf16Kernel() {
        DirectForwardMetalMatvecRowsGatedFfnKernelPlan plan = plan(FFNActivationType.GELU);

        assertEquals("geglu_matvec_rows_bf16_scalar", plan.pathSuffix(1));
        assertEquals("ffn_geglu_matvec_rows_bf16_scalar", plan.profilerKey(1));
        assertEquals("accept:geglu:native_bf16=true:native_rows=12:variant=scalar",
                plan.acceptDecision(12, 1));
    }

    @Test
    void describesUnknownRowsBf16Variant() {
        DirectForwardMetalMatvecRowsGatedFfnKernelPlan plan = plan(FFNActivationType.GELU);

        assertEquals("geglu_matvec_rows_bf16_unknown", plan.pathSuffix(0));
        assertEquals("ffn_geglu_matvec_rows_bf16_unknown", plan.profilerKey(0));
        assertEquals("accept:geglu:native_bf16=true:native_rows=12:variant=unknown",
                plan.acceptDecision(12, 0));
    }

    @Test
    void rejectsUnsupportedActivation() {
        DirectForwardMetalFfnActivationPlan unsupported = DirectForwardMetalFfnActivationPlan.from(null);

        assertThrows(IllegalArgumentException.class,
                () -> DirectForwardMetalMatvecRowsGatedFfnKernelPlan.from(unsupported));
    }

    private static DirectForwardMetalMatvecRowsGatedFfnKernelPlan plan(FFNActivationType activationType) {
        return DirectForwardMetalMatvecRowsGatedFfnKernelPlan.from(
                DirectForwardMetalFfnActivationPlan.from(activationType));
    }
}
