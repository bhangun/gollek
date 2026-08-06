/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

final class InferenceProfile {
    final String mode;
    final boolean detailed;
    final HostLoadSnapshot hostLoadStart;
    HostLoadSnapshot hostLoadEnd;
    long tokenizeNanos;
    long modelLoadNanos;
    long sessionAllocateNanos;
    long firstTokenNanos;
    long prefillNanos;
    long decodeNanos;
    long samplingNanos;
    long attentionNanos;
    long ffnNanos;
    long logitsProjectionNanos;
    long logitsMaterializationNanos;
    long greedyArgmaxNanos;
    final Map<String, Long> linearNanosByOperation = new LinkedHashMap<>();
    final Map<String, Integer> linearPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> logitsPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> ffnPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> attentionPathCounts = new LinkedHashMap<>();
    final Map<String, Integer> greedyArgmaxPathCounts = new LinkedHashMap<>();
    int decodeSteps;

    InferenceProfile(String mode, boolean detailed) {
        this.mode = mode;
        this.detailed = detailed;
        this.hostLoadStart = HostLoadSnapshot.capture();
    }

    Map<String, Object> metadata(String backend) {
        return metadata(backend, -1, -1);
    }

    /**
     * @param promptTokens     prompt token count for llama.cpp-style bench keys;
     *                         {@code -1} skips bench/tokens
     * @param completionTokens generated token count (excluding EOS stop)
     */
    Map<String, Object> metadata(String backend, int promptTokens, int completionTokens) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("profile_mode", mode);
        metadata.put("profile_backend", backend);
        metadata.put("profile_load_ms", roundMillis(modelLoadNanos));
        if (modelLoadNanos > 0L) {
            metadata.put("bench.load_ms", roundMillis(modelLoadNanos));
        }
        metadata.put("profile_tokenize_ms", roundMillis(tokenizeNanos));
        metadata.put("profile_session_allocate_ms", roundMillis(sessionAllocateNanos));
        if (firstTokenNanos > 0L) {
            metadata.put("profile_ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("bench.ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("bench.cold_ttft_ms", roundMillis(firstTokenNanos));
            metadata.put("profile_engine_ttft_ms", roundMillis(engineFirstTokenNanos()));
            metadata.put("bench.engine_ttft_ms", roundMillis(engineFirstTokenNanos()));
            metadata.put("profile_engine_ttft_excluded_ms",
                    roundMillis(engineFirstTokenExcludedNanos()));
        }
        metadata.put("profile_prefill_ms", roundMillis(prefillNanos));
        metadata.put("profile_decode_ms", roundMillis(decodeNanos));
        metadata.put("profile_sampling_ms", roundMillis(samplingNanos));
        metadata.put("profile_attention_ms", roundMillis(attentionNanos));
        metadata.put("profile_ffn_ms", roundMillis(ffnNanos));
        metadata.put("profile_logits_ms", roundMillis(logitsProjectionNanos));
        metadata.put("profile_logits_materialization_ms", roundMillis(logitsMaterializationNanos));
        metadata.put("profile_argmax_ms", roundMillis(greedyArgmaxNanos));
        putHostLoadMetadata(metadata, hostLoadStart, hostLoadEnd());
        linearNanosByOperation.forEach((operation, nanos) -> metadata
                .put("profile_linear_" + sanitizeMetricKey(operation) + "_ms", roundMillis(nanos)));
        linearPathCounts.forEach((path, count) -> metadata
                .put("profile_linear_path_" + sanitizeMetricKey(path) + "_count", count));
        logitsPathCounts.forEach((path, count) -> metadata
                .put("profile_logits_path_" + sanitizeMetricKey(path) + "_count", count));
        ffnPathCounts.forEach((path, count) -> metadata
                .put("profile_ffn_path_" + sanitizeMetricKey(path) + "_count", count));
        attentionPathCounts.forEach((path, count) -> metadata
                .put("profile_attention_path_" + sanitizeMetricKey(path) + "_count", count));
        greedyArgmaxPathCounts.forEach((path, count) -> metadata
                .put("profile_argmax_path_" + sanitizeMetricKey(path) + "_count", count));
        putPathStatusMetadata(metadata);
        FfnStrategyEvidence.putMetadata(metadata, ffnPathCounts);
        putBottleneckMetadata(metadata);
        metadata.put("profile_decode_steps", decodeSteps);
        metadata.put("profile_summary", summary(backend));
        if (promptTokens >= 0) {
            metadata.put("tokens.input", promptTokens);
            metadata.put("tokens.output", Math.max(0, completionTokens));
            metadata.put("tokens.decode", Math.max(0, decodeSteps));
            double prefillSec = prefillNanos / 1_000_000_000.0;
            double decodeSec = decodeNanos / 1_000_000_000.0;
            if (promptTokens > 0 && prefillSec > 1e-12) {
                metadata.put("bench.prefill_tps", promptTokens / prefillSec);
                metadata.put("bench.prefill_ms_per_token", roundMillis(prefillNanos) / promptTokens);
            }
            if (completionTokens > 0 && decodeSec > 1e-12) {
                metadata.put("bench.generation_tps", completionTokens / decodeSec);
            }
            if (decodeSteps > 0 && decodeNanos > 0L) {
                double decodeMsPerToken = roundMillis(decodeNanos) / decodeSteps;
                metadata.put("bench.decode_tps", decodeSteps / decodeSec);
                metadata.put("bench.decode_ms_per_token", decodeMsPerToken);
                metadata.put("bench.tpot_ms", decodeMsPerToken);
            }
        }
        return metadata;
    }

    String summary(String backend) {
        return String.format(Locale.ROOT,
                "backend=%s mode=%s load=%.2fms tokenize=%.2fms session=%.2fms ttft=%.2fms engine_ttft=%.2fms prefill=%.2fms decode=%.2fms tpot=%.2fms sampling=%.2fms argmax=%.2fms attention=%.2fms ffn=%.2fms logits=%.2fms logits_copy=%.2fms steps=%d%s%s%s%s%s%s%s%s%s%s",
                backend, mode,
                roundMillis(modelLoadNanos),
                roundMillis(tokenizeNanos),
                roundMillis(sessionAllocateNanos),
                roundMillis(firstTokenNanos),
                roundMillis(engineFirstTokenNanos()),
                roundMillis(prefillNanos),
                roundMillis(decodeNanos),
                decodeSteps > 0 ? roundMillis(decodeNanos) / decodeSteps : 0.0,
                roundMillis(samplingNanos),
                roundMillis(greedyArgmaxNanos),
                roundMillis(attentionNanos),
                roundMillis(ffnNanos),
                roundMillis(logitsProjectionNanos),
                roundMillis(logitsMaterializationNanos),
                decodeSteps,
                hostLoadSummarySuffix(),
                linearSummarySuffix(),
                linearPathSummarySuffix(),
                logitsPathSummarySuffix(),
                ffnPathSummarySuffix(),
                ffnStrategySummarySuffix(),
                attentionPathSummarySuffix(),
                greedyArgmaxPathSummarySuffix(),
                pathStatusSummarySuffix(),
                pathCoverageSummarySuffix());
    }

    private static double roundMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private long engineFirstTokenNanos() {
        if (firstTokenNanos <= 0L) {
            return 0L;
        }
        return Math.max(0L, firstTokenNanos - engineFirstTokenExcludedNanos());
    }

    private long engineFirstTokenExcludedNanos() {
        return Math.max(0L, modelLoadNanos)
                + Math.max(0L, tokenizeNanos)
                + Math.max(0L, sessionAllocateNanos);
    }

    private String linearSummarySuffix() {
        if (linearNanosByOperation.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" linear={");
        linearNanosByOperation.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder()))
                .limit(8)
                .forEachOrdered(entry -> {
                    if (sb.length() > " linear={".length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey())
                            .append('=')
                            .append(String.format(Locale.ROOT, "%.2f", roundMillis(entry.getValue())));
                });
        sb.append('}');
        return sb.toString();
    }

    private String linearPathSummarySuffix() {
        return pathSummarySuffix("linear_paths", linearPathCounts, 16);
    }

    private String logitsPathSummarySuffix() {
        return pathSummarySuffix("logits_paths", logitsPathCounts, 8);
    }

    private String ffnPathSummarySuffix() {
        return pathSummarySuffix("ffn_paths", ffnPathCounts, 16);
    }

    private String ffnStrategySummarySuffix() {
        return FfnStrategyEvidence.summarySuffix(ffnPathCounts);
    }

    private HostLoadSnapshot hostLoadEnd() {
        if (hostLoadEnd == null) {
            hostLoadEnd = HostLoadSnapshot.capture();
        }
        return hostLoadEnd;
    }

    private String hostLoadSummarySuffix() {
        HostLoadSnapshot end = hostLoadEnd();
        if (!hostLoadStart.isAvailable() && !end.isAvailable()) {
            return "";
        }
        return String.format(Locale.ROOT,
                " host={load=%.2f/%.2f,cpu=%.0f%%,proc=%.0f%%,pressure=%s}",
                hostLoadStart.normalizedLoad(),
                end.normalizedLoad(),
                end.percentOrZero(end.systemCpuLoad()),
                end.percentOrZero(end.processCpuLoad()),
                end.pressureLabel());
    }

    private String attentionPathSummarySuffix() {
        return pathSummarySuffix("attention_paths", attentionPathCounts, 8);
    }

    private String greedyArgmaxPathSummarySuffix() {
        return pathSummarySuffix("argmax_paths", greedyArgmaxPathCounts, 4);
    }

    private String pathStatusSummarySuffix() {
        ProfilePathEvidence.Status linear = ProfilePathEvidence.classify(linearPathCounts);
        ProfilePathEvidence.Status logits = ProfilePathEvidence.classify(logitsPathCounts);
        ProfilePathEvidence.Status ffn = ProfilePathEvidence.classify(ffnPathCounts);
        ProfilePathEvidence.Status attention = ProfilePathEvidence.classify(attentionPathCounts);
        ProfilePathEvidence.Status argmax = ProfilePathEvidence.classify(greedyArgmaxPathCounts);
        ProfilePathEvidence.Status core = ProfilePathEvidence.combine(linear, logits, ffn, attention);
        return " path_status={core=" + core.label()
                + ", linear=" + linear.label()
                + ", logits=" + logits.label()
                + ", ffn=" + ffn.label()
                + ", attention=" + attention.label()
                + ", argmax=" + argmax.label()
                + "}";
    }

    private String pathCoverageSummarySuffix() {
        ProfilePathEvidence.Coverage linear = ProfilePathEvidence.coverage(linearPathCounts);
        ProfilePathEvidence.Coverage logits = ProfilePathEvidence.coverage(logitsPathCounts);
        ProfilePathEvidence.Coverage ffn = ProfilePathEvidence.coverage(ffnPathCounts);
        ProfilePathEvidence.Coverage attention = ProfilePathEvidence.coverage(attentionPathCounts);
        ProfilePathEvidence.Coverage argmax = ProfilePathEvidence.coverage(greedyArgmaxPathCounts);
        ProfilePathEvidence.Coverage core = ProfilePathEvidence.combine(linear, logits, ffn, attention);
        return " path_coverage={core=" + core.label()
                + ", linear=" + linear.label()
                + ", logits=" + logits.label()
                + ", ffn=" + ffn.label()
                + ", attention=" + attention.label()
                + ", argmax=" + argmax.label()
                + "}";
    }

    private void putPathStatusMetadata(Map<String, Object> metadata) {
        ProfilePathEvidence.Status linear = ProfilePathEvidence.classify(linearPathCounts);
        ProfilePathEvidence.Status logits = ProfilePathEvidence.classify(logitsPathCounts);
        ProfilePathEvidence.Status ffn = ProfilePathEvidence.classify(ffnPathCounts);
        ProfilePathEvidence.Status attention = ProfilePathEvidence.classify(attentionPathCounts);
        ProfilePathEvidence.Status argmax = ProfilePathEvidence.classify(greedyArgmaxPathCounts);
        metadata.put("profile_core_path_status",
                ProfilePathEvidence.combine(linear, logits, ffn, attention).label());
        metadata.put("profile_linear_path_status", linear.label());
        metadata.put("profile_logits_path_status", logits.label());
        metadata.put("profile_ffn_path_status", ffn.label());
        metadata.put("profile_attention_path_status", attention.label());
        metadata.put("profile_argmax_path_status", argmax.label());
        putCoverageMetadata(metadata, "core", ProfilePathEvidence.combine(
                ProfilePathEvidence.coverage(linearPathCounts),
                ProfilePathEvidence.coverage(logitsPathCounts),
                ProfilePathEvidence.coverage(ffnPathCounts),
                ProfilePathEvidence.coverage(attentionPathCounts)));
        putCoverageMetadata(metadata, "linear", ProfilePathEvidence.coverage(linearPathCounts));
        putCoverageMetadata(metadata, "logits", ProfilePathEvidence.coverage(logitsPathCounts));
        putCoverageMetadata(metadata, "ffn", ProfilePathEvidence.coverage(ffnPathCounts));
        putCoverageMetadata(metadata, "attention", ProfilePathEvidence.coverage(attentionPathCounts));
        putCoverageMetadata(metadata, "argmax", ProfilePathEvidence.coverage(greedyArgmaxPathCounts));
    }

    private static void putCoverageMetadata(Map<String, Object> metadata, String prefix,
            ProfilePathEvidence.Coverage coverage) {
        metadata.put("profile_" + prefix + "_metal_path_count", coverage.metal());
        metadata.put("profile_" + prefix + "_fallback_path_count", coverage.fallback());
        metadata.put("profile_" + prefix + "_unknown_path_count", coverage.unknown());
        metadata.put("profile_" + prefix + "_total_path_count", coverage.total());
        metadata.put("profile_" + prefix + "_metal_path_ratio", coverage.metalRatio());
    }

    private void putBottleneckMetadata(Map<String, Object> metadata) {
        long denominator = engineFirstTokenNanos();
        if (denominator <= 0L) {
            denominator = prefillNanos + decodeNanos + samplingNanos;
        }
        Bottleneck bottleneck = primaryBottleneck();
        if (denominator <= 0L || bottleneck == null || bottleneck.nanos() <= 0L) {
            return;
        }
        double sharePercent = bottleneck.nanos() * 100.0 / denominator;
        metadata.put("profile_bottleneck_stage", bottleneck.stage());
        metadata.put("profile_bottleneck_value_ms", roundMillis(bottleneck.nanos()));
        metadata.put("profile_bottleneck_share_percent", sharePercent);
        String advice = bottleneckAdvice(bottleneck.stage());
        if (advice != null && !advice.isBlank()) {
            metadata.put("profile_bottleneck_advice", advice);
        }
    }

    private Bottleneck primaryBottleneck() {
        Bottleneck best = null;
        best = Bottleneck.max(best, "ffn", ffnNanos);
        best = Bottleneck.max(best, "attention", attentionNanos);
        best = Bottleneck.max(best, "logits", logitsProjectionNanos);
        best = Bottleneck.max(best, "logits_copy", logitsMaterializationNanos);
        best = Bottleneck.max(best, "decode", decodeNanos);
        best = Bottleneck.max(best, "sampling", samplingNanos);
        if (best == null || best.nanos() <= 0L) {
            best = Bottleneck.max(best, "prefill", prefillNanos);
        }
        return best;
    }

    private String bottleneckAdvice(String stage) {
        return switch (stage) {
            case "ffn" -> ffnBottleneckAdvice();
            case "attention" -> "attention dominates TTFT; inspect FlashAttention gather/projection fusion before FFN tuning";
            case "logits" -> "logits projection dominates TTFT; prioritize vocab projection and last-token logits path";
            case "logits_copy" -> "logits materialization dominates TTFT; avoid full logits host copies on the hot path";
            case "decode" -> "decode dominates the profile; inspect KV-cache and per-token matvec paths";
            case "sampling" -> "sampling dominates the profile; use greedy/native argmax when benchmarking runner throughput";
            case "prefill" -> "prefill dominates TTFT; enable detailed stage profiling to split attention, FFN, and logits";
            default -> null;
        };
    }

    private String ffnBottleneckAdvice() {
        return FfnStrategyEvidence.bottleneckAdvice(ffnPathCounts);
    }

    private static String pathSummarySuffix(String label, Map<String, Integer> counts, int limit) {
        if (counts.isEmpty()) {
            return "";
        }
        String prefix = " " + label + "={";
        StringBuilder sb = new StringBuilder(prefix);
        counts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue(Comparator.reverseOrder()))
                .limit(limit)
                .forEachOrdered(entry -> {
                    if (sb.length() > prefix.length()) {
                        sb.append(", ");
                    }
                    sb.append(entry.getKey()).append('=').append(entry.getValue());
                });
        sb.append('}');
        return sb.toString();
    }

    private static void putHostLoadMetadata(Map<String, Object> metadata,
                                            HostLoadSnapshot start,
                                            HostLoadSnapshot end) {
        if (metadata == null || start == null || end == null) {
            return;
        }
        metadata.put("host.available_processors", end.availableProcessors());
        metadata.put("host.load_average_start", start.systemLoadAverage());
        metadata.put("host.load_average_end", end.systemLoadAverage());
        metadata.put("host.load_average_per_cpu_start", start.normalizedLoad());
        metadata.put("host.load_average_per_cpu_end", end.normalizedLoad());
        metadata.put("host.cpu_load_percent", end.percentOrZero(end.systemCpuLoad()));
        metadata.put("host.process_cpu_load_percent", end.percentOrZero(end.processCpuLoad()));
        if (end.freeMemoryBytes() > 0L) {
            metadata.put("host.memory_free_mb", end.freeMemoryBytes() / (1024.0 * 1024.0));
        }
        if (end.totalMemoryBytes() > 0L) {
            metadata.put("host.memory_total_mb", end.totalMemoryBytes() / (1024.0 * 1024.0));
        }
        if (!Double.isNaN(end.freeMemoryPercent())) {
            metadata.put("host.memory_free_percent", end.freeMemoryPercent() * 100.0);
        }
        metadata.put("host.pressure", end.pressureLabel());
        metadata.put("host.pressure_warning", end.hasPressure());
    }

    private static String sanitizeMetricKey(String raw) {
        StringBuilder out = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char ch = raw.charAt(i);
            if ((ch >= 'a' && ch <= 'z')
                    || (ch >= 'A' && ch <= 'Z')
                    || (ch >= '0' && ch <= '9')) {
                out.append(Character.toLowerCase(ch));
            } else {
                out.append('_');
            }
        }
        return out.toString();
    }

    private record Bottleneck(String stage, long nanos) {
        static Bottleneck max(Bottleneck current, String stage, long nanos) {
            if (nanos <= 0L) {
                return current;
            }
            if (current == null || nanos > current.nanos()) {
                return new Bottleneck(stage, nanos);
            }
            return current;
        }
    }
}
