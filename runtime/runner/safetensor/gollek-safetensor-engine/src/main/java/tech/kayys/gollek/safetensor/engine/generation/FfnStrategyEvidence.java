/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Map;

/**
 * Classifies FFN path counters into stable, machine-readable strategy labels
 * for benchmark gates and profile advice.
 */
final class FfnStrategyEvidence {
    private static final String ROW_PREFILL_ACCEPT_PREFIX = "matvec-gated-ffn-prefill-rows:accept";
    private static final String ROW_PREFILL_FUSED_SKIP_PREFIX =
            "matvec-gated-ffn-prefill-rows:skip:strategy_prefers_fused_geglu_prefill";
    private static final String FUSED_GEGLU_ACCEPT_PREFIX = "fused-gated-ffn:accept:geglu:native_bf16=true";
    private static final String ROW_PREFILL_NATIVE_ROWS_MARKER = ":native_rows=";
    private static final String ROW_PREFILL_VARIANT_MARKER = ":variant=";
    static final String ROW_PREFILL_AB_TEST_PROPERTY =
            "gollek.safetensor.prefer_metal_matvec_ffn_prefill_rows";

    private FfnStrategyEvidence() {
    }

    static Strategy classify(Map<String, Integer> ffnPathCounts) {
        if (containsPrefix(ffnPathCounts, ROW_PREFILL_ACCEPT_PREFIX)) {
            return Strategy.ROW_PREFILL_MATVEC_ACTIVE;
        }
        if (containsPrefix(ffnPathCounts, ROW_PREFILL_FUSED_SKIP_PREFIX)
                && containsPrefix(ffnPathCounts, FUSED_GEGLU_ACCEPT_PREFIX)) {
            return Strategy.FUSED_GEGLU_PREFILL_OVER_ROW_PREFILL;
        }
        if (containsPrefix(ffnPathCounts, FUSED_GEGLU_ACCEPT_PREFIX)) {
            return Strategy.FUSED_GEGLU_PREFILL_ACTIVE;
        }
        return Strategy.UNKNOWN;
    }

    static void putMetadata(Map<String, Object> metadata, Map<String, Integer> ffnPathCounts) {
        Strategy strategy = classify(ffnPathCounts);
        metadata.put("profile_ffn_strategy", strategy.label());
        metadata.put("profile_ffn_strategy_row_prefill_active", strategy.rowPrefillActive());
        metadata.put("profile_ffn_strategy_fused_prefill_active", strategy.fusedPrefillActive());
        if (strategy == Strategy.ROW_PREFILL_MATVEC_ACTIVE) {
            putRowPrefillMetadata(metadata, ffnPathCounts);
        }
        if (strategy == Strategy.FUSED_GEGLU_PREFILL_OVER_ROW_PREFILL) {
            metadata.put("profile_ffn_strategy_ab_test_property", ROW_PREFILL_AB_TEST_PROPERTY);
        }
    }

    static String summarySuffix(Map<String, Integer> ffnPathCounts) {
        Strategy strategy = classify(ffnPathCounts);
        return strategy == Strategy.UNKNOWN ? "" : " ffn_strategy=" + strategy.label();
    }

    static String bottleneckAdvice(Map<String, Integer> ffnPathCounts) {
        return switch (classify(ffnPathCounts)) {
            case ROW_PREFILL_MATVEC_ACTIVE ->
                    "row-prefill matvec FFN is active; disable it if TTFT regresses and compare against fused GEGLU prefill";
            case FUSED_GEGLU_PREFILL_OVER_ROW_PREFILL ->
                    "FFN prefill dominates; fused GEGLU was selected over row-prefill matvec. "
                            + "Use -D" + ROW_PREFILL_AB_TEST_PROPERTY + "=true only for A/B tests";
            case FUSED_GEGLU_PREFILL_ACTIVE ->
                    "FFN prefill dominates; keep fused GEGLU BF16 enabled and prioritize a batched native FFN kernel";
            case UNKNOWN -> "FFN dominates TTFT; inspect gated projection/down projection fusion and FFN weight format";
        };
    }

    private static boolean containsPrefix(Map<String, Integer> pathCounts, String prefix) {
        return pathCounts != null && pathCounts.keySet().stream().anyMatch(path -> path.startsWith(prefix));
    }

    private static void putRowPrefillMetadata(Map<String, Object> metadata, Map<String, Integer> pathCounts) {
        String path = strongestPathWithPrefix(pathCounts, ROW_PREFILL_ACCEPT_PREFIX);
        if (path == null) {
            return;
        }
        Integer nativeRows = extractIntegerValue(path, ROW_PREFILL_NATIVE_ROWS_MARKER);
        if (nativeRows != null) {
            metadata.put("profile_ffn_strategy_row_prefill_native_rows", nativeRows);
        }
        String variant = extractStringValue(path, ROW_PREFILL_VARIANT_MARKER);
        if (variant != null && !variant.isBlank()) {
            metadata.put("profile_ffn_strategy_row_prefill_variant", variant);
        }
    }

    private static String strongestPathWithPrefix(Map<String, Integer> pathCounts, String prefix) {
        if (pathCounts == null || pathCounts.isEmpty()) {
            return null;
        }
        String bestPath = null;
        int bestCount = Integer.MIN_VALUE;
        for (Map.Entry<String, Integer> entry : pathCounts.entrySet()) {
            String path = entry.getKey();
            if (path == null || !path.startsWith(prefix)) {
                continue;
            }
            int count = Math.max(0, entry.getValue() == null ? 0 : entry.getValue());
            if (bestPath == null || count > bestCount) {
                bestPath = path;
                bestCount = count;
            }
        }
        return bestPath;
    }

    private static Integer extractIntegerValue(String path, String marker) {
        String value = extractStringValue(path, marker);
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static String extractStringValue(String path, String marker) {
        if (path == null || marker == null || marker.isEmpty()) {
            return null;
        }
        int start = path.indexOf(marker);
        if (start < 0) {
            return null;
        }
        int valueStart = start + marker.length();
        int valueEnd = path.indexOf(':', valueStart);
        return valueEnd < 0 ? path.substring(valueStart) : path.substring(valueStart, valueEnd);
    }

    enum Strategy {
        ROW_PREFILL_MATVEC_ACTIVE("row_prefill_matvec_active", true, false),
        FUSED_GEGLU_PREFILL_OVER_ROW_PREFILL("fused_geglu_prefill_over_row_prefill", false, true),
        FUSED_GEGLU_PREFILL_ACTIVE("fused_geglu_prefill_active", false, true),
        UNKNOWN("unknown", false, false);

        private final String label;
        private final boolean rowPrefillActive;
        private final boolean fusedPrefillActive;

        Strategy(String label, boolean rowPrefillActive, boolean fusedPrefillActive) {
            this.label = label;
            this.rowPrefillActive = rowPrefillActive;
            this.fusedPrefillActive = fusedPrefillActive;
        }

        String label() {
            return label;
        }

        boolean rowPrefillActive() {
            return rowPrefillActive;
        }

        boolean fusedPrefillActive() {
            return fusedPrefillActive;
        }
    }
}
