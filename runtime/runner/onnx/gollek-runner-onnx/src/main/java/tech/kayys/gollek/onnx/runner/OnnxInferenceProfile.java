package tech.kayys.gollek.onnx.runner;

import java.util.Map;

import tech.kayys.gollek.spi.inference.InferenceRequest;

final class OnnxInferenceProfile {

    private static final String PROFILE_PROPERTY = "gollek.profile";

    private final boolean enabled;
    private final long requestStartNanos;
    private long tokenizeNanos;
    private long inputPrepareNanos;
    private long prefillRunNanos;
    private long decodeRunNanos;
    private long logitsSelectNanos;
    private long samplingNanos;
    private long finalDecodeNanos;
    private long firstTokenNanos;
    private int prefillSteps;
    private int decodeSteps;
    private boolean workspaceLeaseRecorded;
    private boolean workspaceReused;
    private int workspaceEvictions;
    private int workspaceRequestedCapacity;
    private int workspaceCapacity;
    private int scalarInputIdsCacheHits;
    private int scalarInputIdsCacheMisses;
    private int scalarPositionIdsCacheHits;
    private int scalarPositionIdsCacheMisses;
    private int attentionMaskCacheHits;
    private int attentionMaskCacheMisses;
    private int prefixInputIdsCacheHits;
    private int prefixInputIdsCacheMisses;
    private int rangePositionIdsCacheHits;
    private int rangePositionIdsCacheMisses;
    private int inputTensorCacheEvictions;

    private OnnxInferenceProfile(boolean enabled, long requestStartNanos) {
        this.enabled = enabled;
        this.requestStartNanos = requestStartNanos;
    }

    static OnnxInferenceProfile start(InferenceRequest request) {
        return new OnnxInferenceProfile(enabledFor(request), System.nanoTime());
    }

    boolean enabled() {
        return enabled;
    }

    long mark() {
        return enabled ? System.nanoTime() : 0L;
    }

    void recordTokenize(long startNanos) {
        tokenizeNanos += elapsed(startNanos);
    }

    void recordInputPrepare(long startNanos) {
        inputPrepareNanos += elapsed(startNanos);
    }

    void recordOrtRun(long startNanos, boolean prefill) {
        long nanos = elapsed(startNanos);
        if (prefill) {
            prefillRunNanos += nanos;
            prefillSteps++;
        } else {
            decodeRunNanos += nanos;
            decodeSteps++;
        }
    }

    void recordLogitsSelect(long startNanos) {
        logitsSelectNanos += elapsed(startNanos);
    }

    void recordSampling(long startNanos) {
        samplingNanos += elapsed(startNanos);
    }

    void recordFinalDecode(long startNanos) {
        finalDecodeNanos += elapsed(startNanos);
    }

    void markFirstToken() {
        if (enabled && firstTokenNanos <= 0L) {
            firstTokenNanos = System.nanoTime() - requestStartNanos;
        }
    }

    void recordWorkspaceLease(boolean reused, int evictions, int requestedCapacity, int capacity) {
        if (!enabled) {
            return;
        }
        workspaceLeaseRecorded = true;
        workspaceReused = reused;
        workspaceEvictions = Math.max(0, evictions);
        workspaceRequestedCapacity = Math.max(0, requestedCapacity);
        workspaceCapacity = Math.max(0, capacity);
    }

    void recordInputTensorCache(OnnxInputTensorCacheStats stats) {
        if (!enabled || stats == null) {
            return;
        }
        scalarInputIdsCacheHits += Math.max(0, stats.inputIdsHits());
        scalarInputIdsCacheMisses += Math.max(0, stats.inputIdsMisses());
        scalarPositionIdsCacheHits += Math.max(0, stats.positionIdsHits());
        scalarPositionIdsCacheMisses += Math.max(0, stats.positionIdsMisses());
        attentionMaskCacheHits += Math.max(0, stats.attentionMaskHits());
        attentionMaskCacheMisses += Math.max(0, stats.attentionMaskMisses());
        prefixInputIdsCacheHits += Math.max(0, stats.prefixInputIdsHits());
        prefixInputIdsCacheMisses += Math.max(0, stats.prefixInputIdsMisses());
        rangePositionIdsCacheHits += Math.max(0, stats.rangePositionIdsHits());
        rangePositionIdsCacheMisses += Math.max(0, stats.rangePositionIdsMisses());
        inputTensorCacheEvictions += Math.max(0, stats.evictions());
    }

    Map<String, Object> metadata(String backend, int inputTokens, int outputTokens, long totalDurationMs) {
        if (!enabled) {
            return Map.of();
        }
        return snapshot(backend, inputTokens, outputTokens, totalDurationMs).metadata();
    }

    OnnxInferenceProfileSnapshot snapshot(String backend, int inputTokens, int outputTokens, long totalDurationMs) {
        return new OnnxInferenceProfileSnapshot(
                backend,
                inputTokens,
                outputTokens,
                totalDurationMs,
                tokenizeNanos,
                inputPrepareNanos,
                prefillRunNanos,
                decodeRunNanos,
                logitsSelectNanos,
                samplingNanos,
                finalDecodeNanos,
                firstTokenNanos,
                prefillSteps,
                decodeSteps,
                workspaceLeaseRecorded,
                workspaceReused,
                workspaceEvictions,
                workspaceRequestedCapacity,
                workspaceCapacity,
                scalarInputIdsCacheHits,
                scalarInputIdsCacheMisses,
                scalarPositionIdsCacheHits,
                scalarPositionIdsCacheMisses,
                attentionMaskCacheHits,
                attentionMaskCacheMisses,
                prefixInputIdsCacheHits,
                prefixInputIdsCacheMisses,
                rangePositionIdsCacheHits,
                rangePositionIdsCacheMisses,
                inputTensorCacheEvictions);
    }

    private long elapsed(long startNanos) {
        if (!enabled || startNanos <= 0L) {
            return 0L;
        }
        return Math.max(0L, System.nanoTime() - startNanos);
    }

    private static boolean enabledFor(InferenceRequest request) {
        return Boolean.getBoolean(PROFILE_PROPERTY)
                || truthy(request == null ? null : request.getParameters().get("profile"))
                || truthy(request == null ? null : request.getParameters().get("onnx_profile"))
                || truthy(request == null ? null : request.getParameters().get("bench"))
                || truthy(request == null ? null : request.getParameters().get("benchmark"))
                || truthy(request == null ? null : request.getMetadata().get("profile"))
                || truthy(request == null ? null : request.getMetadata().get("onnx_profile"));
    }

    private static boolean truthy(Object value) {
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof Number number) {
            return number.intValue() != 0;
        }
        if (value instanceof String text) {
            String normalized = text.trim().toLowerCase(java.util.Locale.ROOT);
            return normalized.equals("true")
                    || normalized.equals("1")
                    || normalized.equals("yes")
                    || normalized.equals("on")
                    || normalized.equals("bench")
                    || normalized.equals("benchmark");
        }
        return false;
    }
}
