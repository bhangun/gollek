/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.model.FFNActivationType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalFfnActivationPlanTest {

    @Test
    void describesSiluNamesAndProfilerKeys() {
        DirectForwardMetalFfnActivationPlan plan =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.SILU);

        assertTrue(plan.supported());
        assertTrue(plan.silu());
        assertFalse(plan.gelu());
        assertEquals("swiglu", plan.metalName());
        assertEquals("accept:swiglu:native_bf16=true", plan.acceptDecision(true));
        assertEquals("swiglu_fused_f16", plan.fusedPathSuffix(false));
        assertEquals("swiglu_matvec_bf16", plan.matvecPathSuffix(true));
        assertEquals("swiglu_gate_up_matvec_f16", plan.gateUpPathSuffix(false));
        assertEquals("ffn_fused_metal", plan.fusedProfilerKey());
        assertEquals("ffn_swiglu_matvec_metal", plan.matvecProfilerKey(false));
        assertEquals("ffn_swiglu_matvec_bf16", plan.matvecProfilerKey(true));
    }

    @Test
    void describesGeluNamesAndProfilerKeys() {
        DirectForwardMetalFfnActivationPlan plan =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.GELU);

        assertTrue(plan.supported());
        assertFalse(plan.silu());
        assertTrue(plan.gelu());
        assertEquals("geglu", plan.metalName());
        assertEquals("accept:geglu:native_bf16=false", plan.acceptDecision(false));
        assertEquals("geglu_fused_bf16", plan.fusedPathSuffix(true));
        assertEquals("geglu_matvec_f16", plan.matvecPathSuffix(false));
        assertEquals("geglu_gate_up_matvec_bf16", plan.gateUpPathSuffix(true));
        assertEquals("ffn_geglu_fused_metal", plan.fusedProfilerKey());
        assertEquals("ffn_geglu_matvec_metal", plan.matvecProfilerKey(false));
        assertEquals("ffn_geglu_matvec_bf16", plan.matvecProfilerKey(true));
    }

    @Test
    void unsupportedActivationKeepsTraceReasonStable() {
        DirectForwardMetalFfnActivationPlan plan = DirectForwardMetalFfnActivationPlan.from(null);

        assertFalse(plan.supported());
        assertEquals("unsupported_activation:null", plan.unsupportedReason());
        assertThrows(IllegalStateException.class, plan::metalName);
    }

    @Test
    void reportsFusedCapabilityRejectionsByActivation() {
        assertEquals("swiglu_symbol_unavailable",
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.SILU)
                        .fusedHalfCapabilityRejection(DirectForwardMetalCapabilities.EMPTY));
        assertEquals("geglu_symbol_unavailable",
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.GELU)
                        .fusedHalfCapabilityRejection(DirectForwardMetalCapabilities.EMPTY));

        assertNull(DirectForwardMetalFfnActivationPlan.from(FFNActivationType.SILU)
                .fusedHalfCapabilityRejection(allCapabilities()));
        assertNull(DirectForwardMetalFfnActivationPlan.from(FFNActivationType.GELU)
                .fusedHalfCapabilityRejection(allCapabilities()));
    }

    @Test
    void reportsMatvecCapabilityRejectionsByActivationAndPrecision() {
        DirectForwardMetalFfnActivationPlan silu =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.SILU);
        DirectForwardMetalFfnActivationPlan gelu =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.GELU);

        assertEquals("swiglu_matvec_symbol_unavailable",
                silu.matvecCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, false));
        assertEquals("swiglu_bf16_matvec_symbol_unavailable",
                silu.matvecCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, true));
        assertEquals("geglu_matvec_symbol_unavailable",
                gelu.matvecCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, false));
        assertEquals("geglu_bf16_matvec_symbol_unavailable",
                gelu.matvecCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, true));

        assertNull(silu.matvecCapabilityRejection(allCapabilities(), false));
        assertNull(silu.matvecCapabilityRejection(allCapabilities(), true));
        assertNull(gelu.matvecCapabilityRejection(allCapabilities(), false));
        assertNull(gelu.matvecCapabilityRejection(allCapabilities(), true));
    }

    @Test
    void reportsGateUpCapabilityRejectionsByActivationAndPrecision() {
        DirectForwardMetalFfnActivationPlan silu =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.SILU);
        DirectForwardMetalFfnActivationPlan gelu =
                DirectForwardMetalFfnActivationPlan.from(FFNActivationType.GELU);

        assertEquals("swiglu_symbol_unavailable",
                silu.gateUpCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, false));
        assertEquals("swiglu_bf16_symbol_unavailable",
                silu.gateUpCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, true));
        assertEquals("geglu_symbol_unavailable",
                gelu.gateUpCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, false));
        assertEquals("geglu_bf16_symbol_unavailable",
                gelu.gateUpCapabilityRejection(DirectForwardMetalCapabilities.EMPTY, true));

        assertNull(silu.gateUpCapabilityRejection(allCapabilities(), false));
        assertNull(silu.gateUpCapabilityRejection(allCapabilities(), true));
        assertNull(gelu.gateUpCapabilityRejection(allCapabilities(), false));
        assertNull(gelu.gateUpCapabilityRejection(allCapabilities(), true));
    }

    private static DirectForwardMetalCapabilities allCapabilities() {
        return new DirectForwardMetalCapabilities(
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, false, false);
    }
}
