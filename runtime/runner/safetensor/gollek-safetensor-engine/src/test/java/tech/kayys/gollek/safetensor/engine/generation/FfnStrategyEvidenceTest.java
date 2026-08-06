/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FfnStrategyEvidenceTest {

    @Test
    void classifiesRowPrefillMatvecAcceptanceFirst() {
        Map<String, Integer> paths = paths(
                "matvec-gated-ffn-prefill-rows:accept:geglu:native_bf16=true:native_rows=12:variant=x4");
        Map<String, Object> metadata = new LinkedHashMap<>();

        assertEquals(
                FfnStrategyEvidence.Strategy.ROW_PREFILL_MATVEC_ACTIVE,
                FfnStrategyEvidence.classify(paths));
        FfnStrategyEvidence.putMetadata(metadata, paths);
        assertEquals("row_prefill_matvec_active", metadata.get("profile_ffn_strategy"));
        assertEquals(true, metadata.get("profile_ffn_strategy_row_prefill_active"));
        assertEquals(false, metadata.get("profile_ffn_strategy_fused_prefill_active"));
        assertEquals(12, metadata.get("profile_ffn_strategy_row_prefill_native_rows"));
        assertEquals("x4", metadata.get("profile_ffn_strategy_row_prefill_variant"));
        assertEquals(
                "row-prefill matvec FFN is active; disable it if TTFT regresses and compare against fused GEGLU prefill",
                FfnStrategyEvidence.bottleneckAdvice(paths));
    }

    @Test
    void usesStrongestAcceptedRowPrefillPathForVariantMetadata() {
        Map<String, Integer> paths = new LinkedHashMap<>();
        paths.put("matvec-gated-ffn-prefill-rows:accept:geglu:native_bf16=true:native_rows=12:variant=scalar", 3);
        paths.put("matvec-gated-ffn-prefill-rows:accept:geglu:native_bf16=true:native_rows=16:variant=x4", 9);
        Map<String, Object> metadata = new LinkedHashMap<>();

        FfnStrategyEvidence.putMetadata(metadata, paths);

        assertEquals(16, metadata.get("profile_ffn_strategy_row_prefill_native_rows"));
        assertEquals("x4", metadata.get("profile_ffn_strategy_row_prefill_variant"));
    }

    @Test
    void classifiesFusedPrefillSelectedOverRowPrefill() {
        Map<String, Integer> paths = paths(
                "matvec-gated-ffn-prefill-rows:skip:strategy_prefers_fused_geglu_prefill",
                "fused-gated-ffn:accept:geglu:native_bf16=true");

        assertEquals(
                FfnStrategyEvidence.Strategy.FUSED_GEGLU_PREFILL_OVER_ROW_PREFILL,
                FfnStrategyEvidence.classify(paths));
        assertEquals(
                "FFN prefill dominates; fused GEGLU was selected over row-prefill matvec. "
                        + "Use -Dgollek.safetensor.prefer_metal_matvec_ffn_prefill_rows=true only for A/B tests",
                FfnStrategyEvidence.bottleneckAdvice(paths));
        assertEquals(
                " ffn_strategy=fused_geglu_prefill_over_row_prefill",
                FfnStrategyEvidence.summarySuffix(paths));
    }

    @Test
    void classifiesPlainFusedPrefillAcceptance() {
        Map<String, Integer> paths = paths("fused-gated-ffn:accept:geglu:native_bf16=true");

        assertEquals(
                FfnStrategyEvidence.Strategy.FUSED_GEGLU_PREFILL_ACTIVE,
                FfnStrategyEvidence.classify(paths));
    }

    @Test
    void classifiesMissingEvidenceAsUnknown() {
        assertEquals(FfnStrategyEvidence.Strategy.UNKNOWN, FfnStrategyEvidence.classify(Map.of()));
        assertEquals("", FfnStrategyEvidence.summarySuffix(Map.of()));
    }

    private static Map<String, Integer> paths(String... keys) {
        Map<String, Integer> paths = new LinkedHashMap<>();
        for (String key : keys) {
            paths.put(key, 1);
        }
        return paths;
    }
}
