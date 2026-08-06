package tech.kayys.gollek.onnx.runner;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

record OnnxInferenceProfileSnapshot(
        String backend,
        int inputTokens,
        int outputTokens,
        long totalDurationMs,
        long tokenizeNanos,
        long inputPrepareNanos,
        long prefillRunNanos,
        long decodeRunNanos,
        long logitsSelectNanos,
        long samplingNanos,
        long finalDecodeNanos,
        long firstTokenNanos,
        int prefillSteps,
        int decodeSteps,
        boolean workspaceLeaseRecorded,
        boolean workspaceReused,
        int workspaceEvictions,
        int workspaceRequestedCapacity,
        int workspaceCapacity,
        int scalarInputIdsCacheHits,
        int scalarInputIdsCacheMisses,
        int scalarPositionIdsCacheHits,
        int scalarPositionIdsCacheMisses,
        int attentionMaskCacheHits,
        int attentionMaskCacheMisses,
        int prefixInputIdsCacheHits,
        int prefixInputIdsCacheMisses,
        int rangePositionIdsCacheHits,
        int rangePositionIdsCacheMisses,
        int inputTensorCacheEvictions) {

    Map<String, Object> metadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("profile_mode", "onnx");
        metadata.put("profile_backend", backend);
        metadata.put("profile_onnx_tokenize_ms", millis(tokenizeNanos));
        metadata.put("profile_onnx_input_prepare_ms", millis(inputPrepareNanos));
        metadata.put("profile_onnx_prefill_run_ms", millis(prefillRunNanos));
        metadata.put("profile_onnx_decode_run_ms", millis(decodeRunNanos));
        metadata.put("profile_onnx_ort_run_ms", millis(ortRunNanos()));
        metadata.put("profile_onnx_logits_select_ms", millis(logitsSelectNanos));
        metadata.put("profile_onnx_sampling_ms", millis(samplingNanos));
        metadata.put("profile_onnx_final_decode_ms", millis(finalDecodeNanos));
        metadata.put("profile_onnx_prefill_steps", prefillSteps);
        metadata.put("profile_onnx_decode_steps", decodeSteps);
        metadata.put("profile_onnx_generation_ms", millis(generationNanos()));
        metadata.put("profile_onnx_profiled_ms", millis(profiledNanos()));
        metadata.put("profile_onnx_unprofiled_ms", unprofiledMs());
        putWorkspaceMetadata(metadata);
        putInputTensorCacheMetadata(metadata);
        putPrimaryStage(metadata);
        metadata.put("profile_onnx_summary", summary());
        putBenchmarkMetadata(metadata);
        return metadata;
    }

    private void putBenchmarkMetadata(Map<String, Object> metadata) {
        metadata.put("tokens.input", Math.max(0, inputTokens));
        metadata.put("tokens.output", Math.max(0, outputTokens));
        metadata.put("tokens.decode", Math.max(0, decodeSteps));
        metadata.put("bench.latency_ms", Math.max(0L, totalDurationMs));
        if (firstTokenNanos > 0L) {
            metadata.put("bench.ttft_ms", millis(firstTokenNanos));
            metadata.put("bench.cold_ttft_ms", millis(firstTokenNanos));
        }
        if (inputTokens > 0 && prefillRunNanos > 0L) {
            double prefillSec = prefillRunNanos / 1_000_000_000.0;
            metadata.put("bench.prefill_tps", inputTokens / prefillSec);
            metadata.put("bench.prefill_ms", millis(prefillRunNanos));
            metadata.put("bench.prefill_ms_per_token", millis(prefillRunNanos) / inputTokens);
        }
        if (decodeSteps > 0 && decodeRunNanos > 0L) {
            double decodeSec = decodeRunNanos / 1_000_000_000.0;
            double decodeMsPerToken = millis(decodeRunNanos) / decodeSteps;
            metadata.put("bench.decode_tps", decodeSteps / decodeSec);
            metadata.put("bench.decode_ms", millis(decodeRunNanos));
            metadata.put("bench.decode_ms_per_token", decodeMsPerToken);
            metadata.put("bench.tpot_ms", decodeMsPerToken);
        }
        if (outputTokens > 0 && generationNanos() > 0L) {
            metadata.put("bench.generation_tps", outputTokens / (generationNanos() / 1_000_000_000.0));
        }
    }

    private void putWorkspaceMetadata(Map<String, Object> metadata) {
        if (!workspaceLeaseRecorded) {
            return;
        }
        metadata.put("profile_onnx_workspace_reused", workspaceReused);
        metadata.put("profile_onnx_workspace_evictions", workspaceEvictions);
        metadata.put("profile_onnx_workspace_requested_capacity", workspaceRequestedCapacity);
        metadata.put("profile_onnx_workspace_capacity", workspaceCapacity);
        metadata.put("profile_onnx_workspace", workspaceSummary());
    }

    private void putInputTensorCacheMetadata(Map<String, Object> metadata) {
        int scalarHits = scalarCacheHits();
        int scalarMisses = scalarCacheMisses();
        metadata.put("profile_onnx_scalar_tensor_cache_hits", scalarHits);
        metadata.put("profile_onnx_scalar_tensor_cache_misses", scalarMisses);
        metadata.put("profile_onnx_input_ids_cache_hits", scalarInputIdsCacheHits);
        metadata.put("profile_onnx_input_ids_cache_misses", scalarInputIdsCacheMisses);
        metadata.put("profile_onnx_position_ids_cache_hits", scalarPositionIdsCacheHits);
        metadata.put("profile_onnx_position_ids_cache_misses", scalarPositionIdsCacheMisses);
        metadata.put("profile_onnx_scalar_tensor_cache_hit_rate_percent", scalarCacheHitRatePercent());
        metadata.put("profile_onnx_scalar_tensor_cache", scalarCacheSummary());
        metadata.put("profile_onnx_attention_mask_cache_hits", attentionMaskCacheHits);
        metadata.put("profile_onnx_attention_mask_cache_misses", attentionMaskCacheMisses);
        metadata.put("profile_onnx_prefix_input_ids_cache_hits", prefixInputIdsCacheHits);
        metadata.put("profile_onnx_prefix_input_ids_cache_misses", prefixInputIdsCacheMisses);
        metadata.put("profile_onnx_range_position_ids_cache_hits", rangePositionIdsCacheHits);
        metadata.put("profile_onnx_range_position_ids_cache_misses", rangePositionIdsCacheMisses);
        metadata.put("profile_onnx_input_tensor_cache_hits", inputTensorCacheHits());
        metadata.put("profile_onnx_input_tensor_cache_misses", inputTensorCacheMisses());
        metadata.put("profile_onnx_input_tensor_cache_evictions", inputTensorCacheEvictions);
        metadata.put("profile_onnx_input_tensor_cache_hit_rate_percent", inputTensorCacheHitRatePercent());
        metadata.put("profile_onnx_input_tensor_cache", inputTensorCacheSummary());
    }

    private void putPrimaryStage(Map<String, Object> metadata) {
        StageSample primary = primaryStage();
        if (profiledNanos() <= 0L || primary.nanos() <= 0L) {
            return;
        }
        metadata.put("profile_onnx_primary_stage", primary.name());
        metadata.put("profile_onnx_primary_value_ms", millis(primary.nanos()));
        metadata.put("profile_onnx_primary_share_percent", primary.nanos() * 100.0 / profiledNanos());
    }

    private StageSample primaryStage() {
        return primaryStage(
                new StageSample("tokenize", tokenizeNanos),
                new StageSample("input_prepare", inputPrepareNanos),
                new StageSample("ort_run", ortRunNanos()),
                new StageSample("logits_select", logitsSelectNanos),
                new StageSample("sampling", samplingNanos),
                new StageSample("final_decode", finalDecodeNanos));
    }

    private static StageSample primaryStage(StageSample... samples) {
        StageSample primary = new StageSample("none", 0L);
        for (StageSample sample : samples) {
            if (sample.nanos() > primary.nanos()) {
                primary = sample;
            }
        }
        return primary;
    }

    private String summary() {
        return String.format(Locale.ROOT,
                "backend=%s tokenize=%.2fms input=%.2fms prefill_run=%.2fms decode_run=%.2fms logits_select=%.2fms sampling=%.2fms final_decode=%.2fms ttft=%.2fms generation=%.2fms profiled=%.2fms workspace=%s input_cache=%s",
                backend,
                millis(tokenizeNanos),
                millis(inputPrepareNanos),
                millis(prefillRunNanos),
                millis(decodeRunNanos),
                millis(logitsSelectNanos),
                millis(samplingNanos),
                millis(finalDecodeNanos),
                millis(firstTokenNanos),
                millis(generationNanos()),
                millis(profiledNanos()),
                workspaceLeaseRecorded ? workspaceSummary() : "unknown",
                inputTensorCacheSummary());
    }

    private String workspaceSummary() {
        return String.format(Locale.ROOT,
                "%s requested=%d capacity=%d evictions=%d",
                workspaceReused ? "reused" : "created",
                workspaceRequestedCapacity,
                workspaceCapacity,
                workspaceEvictions);
    }

    private String scalarCacheSummary() {
        return String.format(Locale.ROOT,
                "hits=%d misses=%d hit_rate=%.1f%%",
                scalarCacheHits(),
                scalarCacheMisses(),
                scalarCacheHitRatePercent());
    }

    private double scalarCacheHitRatePercent() {
        int total = scalarCacheHits() + scalarCacheMisses();
        return total == 0 ? 0.0 : scalarCacheHits() * 100.0 / total;
    }

    private String inputTensorCacheSummary() {
        return String.format(Locale.ROOT,
                "hits=%d misses=%d hit_rate=%.1f%% evictions=%d scalar=%d/%d attention=%d/%d prefix_ids=%d/%d position_ranges=%d/%d",
                inputTensorCacheHits(),
                inputTensorCacheMisses(),
                inputTensorCacheHitRatePercent(),
                inputTensorCacheEvictions,
                scalarCacheHits(),
                scalarCacheMisses(),
                attentionMaskCacheHits,
                attentionMaskCacheMisses,
                prefixInputIdsCacheHits,
                prefixInputIdsCacheMisses,
                rangePositionIdsCacheHits,
                rangePositionIdsCacheMisses);
    }

    private double inputTensorCacheHitRatePercent() {
        int total = inputTensorCacheHits() + inputTensorCacheMisses();
        return total == 0 ? 0.0 : inputTensorCacheHits() * 100.0 / total;
    }

    private int inputTensorCacheHits() {
        return scalarCacheHits() + attentionMaskCacheHits + prefixInputIdsCacheHits + rangePositionIdsCacheHits;
    }

    private int inputTensorCacheMisses() {
        return scalarCacheMisses() + attentionMaskCacheMisses + prefixInputIdsCacheMisses
                + rangePositionIdsCacheMisses;
    }

    private int scalarCacheHits() {
        return scalarInputIdsCacheHits + scalarPositionIdsCacheHits;
    }

    private int scalarCacheMisses() {
        return scalarInputIdsCacheMisses + scalarPositionIdsCacheMisses;
    }

    private long ortRunNanos() {
        return prefillRunNanos + decodeRunNanos;
    }

    private long generationNanos() {
        return inputPrepareNanos + ortRunNanos() + logitsSelectNanos + samplingNanos + finalDecodeNanos;
    }

    private long profiledNanos() {
        return tokenizeNanos + generationNanos();
    }

    private double unprofiledMs() {
        return Math.max(0.0, totalDurationMs - millis(profiledNanos()));
    }

    private static double millis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private record StageSample(String name, long nanos) {
    }
}
