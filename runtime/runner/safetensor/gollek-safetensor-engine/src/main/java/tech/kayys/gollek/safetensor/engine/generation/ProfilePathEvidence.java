/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Locale;
import java.util.Map;

/**
 * Classifies profiler path counters into gate-friendly accelerator evidence.
 */
final class ProfilePathEvidence {
    private ProfilePathEvidence() {
    }

    static Status classify(Map<String, Integer> pathCounts) {
        if (pathCounts == null || pathCounts.isEmpty()) {
            return Status.MISSING;
        }
        boolean metal = false;
        boolean fallback = false;
        boolean unknown = false;
        boolean seen = false;
        for (String path : pathCounts.keySet()) {
            PathKind kind = kindOf(path);
            if (kind == PathKind.IGNORED) {
                continue;
            }
            seen = true;
            metal |= kind == PathKind.METAL;
            fallback |= kind == PathKind.FALLBACK;
            unknown |= kind == PathKind.UNKNOWN;
        }
        if (!seen) {
            return Status.MISSING;
        }
        if (metal && !fallback && !unknown) {
            return Status.METAL;
        }
        if (fallback && !metal && !unknown) {
            return Status.FALLBACK;
        }
        if (unknown && !metal && !fallback) {
            return Status.UNKNOWN;
        }
        return Status.MIXED;
    }

    static Status combine(Status... statuses) {
        boolean seen = false;
        boolean metal = false;
        boolean fallback = false;
        boolean unknown = false;
        boolean missing = false;
        if (statuses == null || statuses.length == 0) {
            return Status.MISSING;
        }
        for (Status status : statuses) {
            Status effective = status == null ? Status.MISSING : status;
            switch (effective) {
                case MISSING -> missing = true;
                case METAL -> {
                    seen = true;
                    metal = true;
                }
                case FALLBACK -> {
                    seen = true;
                    fallback = true;
                }
                case UNKNOWN -> {
                    seen = true;
                    unknown = true;
                }
                case MIXED -> {
                    seen = true;
                    metal = true;
                    fallback = true;
                }
            }
        }
        if (!seen) {
            return Status.MISSING;
        }
        if (missing || (metal && fallback) || (metal && unknown) || (fallback && unknown)) {
            return Status.MIXED;
        }
        if (metal) {
            return Status.METAL;
        }
        if (fallback) {
            return Status.FALLBACK;
        }
        return Status.UNKNOWN;
    }

    static Coverage coverage(Map<String, Integer> pathCounts) {
        if (pathCounts == null || pathCounts.isEmpty()) {
            return Coverage.EMPTY;
        }
        int metal = 0;
        int fallback = 0;
        int unknown = 0;
        for (Map.Entry<String, Integer> entry : pathCounts.entrySet()) {
            int count = Math.max(0, entry.getValue() == null ? 0 : entry.getValue());
            switch (kindOf(entry.getKey())) {
                case METAL -> metal += count;
                case FALLBACK -> fallback += count;
                case UNKNOWN -> unknown += count;
                case IGNORED -> {
                    // Admission probes did not execute work, so they should not
                    // dilute accelerator coverage for the path that actually ran.
                }
            }
        }
        return new Coverage(metal, fallback, unknown);
    }

    static Coverage combine(Coverage... coverages) {
        int metal = 0;
        int fallback = 0;
        int unknown = 0;
        if (coverages == null || coverages.length == 0) {
            return Coverage.EMPTY;
        }
        for (Coverage coverage : coverages) {
            if (coverage == null) {
                continue;
            }
            metal += coverage.metal();
            fallback += coverage.fallback();
            unknown += coverage.unknown();
        }
        return new Coverage(metal, fallback, unknown);
    }

    private static PathKind kindOf(String path) {
        if (path == null || path.isBlank()) {
            return PathKind.UNKNOWN;
        }
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.contains(":reject:") || lower.contains(":skip:")) {
            return PathKind.IGNORED;
        }
        if (lower.contains("metal")
                || lower.contains("mtl")
                || lower.contains("mps")
                || lower.contains("gpu")
                || lower.contains("native_argmax")
                || lower.contains("native_bf16")
                || lower.contains("native_f16")
                || lower.contains("matmul_bf16")
                || lower.contains("matmul_f16")
                || lower.contains("bf16_matvec")
                || lower.contains("f16_matvec")
                || lower.contains("fa4_")
                || lower.endsWith("_fa4")) {
            return PathKind.METAL;
        }
        if (lower.contains("cpu")
                || lower.contains("java")
                || lower.contains("accelerate")
                || lower.contains("fallback")
                || lower.contains("skip")
                || lower.contains("reject")
                || lower.contains("unavailable")) {
            return PathKind.FALLBACK;
        }
        return PathKind.UNKNOWN;
    }

    enum Status {
        MISSING("missing"),
        METAL("metal"),
        FALLBACK("fallback"),
        MIXED("mixed"),
        UNKNOWN("unknown");

        private final String label;

        Status(String label) {
            this.label = label;
        }

        String label() {
            return label;
        }
    }

    record Coverage(int metal, int fallback, int unknown) {
        private static final Coverage EMPTY = new Coverage(0, 0, 0);

        int total() {
            return metal + fallback + unknown;
        }

        double metalRatio() {
            int total = total();
            return total == 0 ? 0.0 : (double) metal / total;
        }

        String label() {
            return metal + "/" + total();
        }
    }

    private enum PathKind {
        METAL,
        FALLBACK,
        UNKNOWN,
        IGNORED
    }
}
