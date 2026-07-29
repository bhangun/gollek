package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Map;

import jakarta.enterprise.inject.Instance;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;

public final class DirectInferenceProfiler {
    private static final String PROFILE_PROPERTY = "gollek.profile";
    private static final long PROFILING_INACTIVE_NANOS = -1L;
    private static final ThreadLocal<InferenceProfile> ACTIVE_PROFILE = new ThreadLocal<>();

    private DirectInferenceProfiler() {
    }

    public static void recordAttentionNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.attentionNanos += nanos;
        }
    }

    public static void recordAttentionPath(String path) {
        if (path == null || path.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.attentionPathCounts.merge(path, 1, Integer::sum);
        }
    }

    public static void recordFfnNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.ffnNanos += nanos;
        }
    }

    public static void recordLogitsProjectionNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.logitsProjectionNanos += nanos;
        }
    }

    public static void recordLogitsMaterializationNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.logitsMaterializationNanos += nanos;
        }
    }

    public static void recordGreedyArgmaxNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.greedyArgmaxNanos += nanos;
        }
    }

    public static void recordGreedyArgmaxPath(String path) {
        if (path == null || path.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.greedyArgmaxPathCounts.merge(path, 1, Integer::sum);
        }
    }

    static long startGreedyArgmaxTiming() {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        return profile != null && profile.detailed ? System.nanoTime() : PROFILING_INACTIVE_NANOS;
    }

    static void recordGreedyArgmaxTiming(long startedNanos, String path) {
        if (startedNanos < 0L || path == null || path.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.greedyArgmaxNanos += System.nanoTime() - startedNanos;
            profile.greedyArgmaxPathCounts.merge(path, 1, Integer::sum);
        }
    }

    public static void recordLinearNanos(String operation, long nanos) {
        if (operation == null || operation.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.linearNanosByOperation.merge(operation, nanos, Long::sum);
        }
    }

    public static void recordLinearPath(String operation, String path) {
        if (operation == null || operation.isBlank() || path == null || path.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.linearPathCounts.merge(operation + ":" + path, 1, Integer::sum);
            if ("logits".equals(operation)) {
                profile.logitsPathCounts.merge(path, 1, Integer::sum);
            }
        }
    }

    public static void recordFfnPath(String path) {
        if (path == null || path.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.ffnPathCounts.merge(path, 1, Integer::sum);
        }
    }

    static void runWithProfileSuspended(Runnable action) {
        InferenceProfile active = ACTIVE_PROFILE.get();
        if (active == null) {
            action.run();
            return;
        }
        ACTIVE_PROFILE.remove();
        try {
            action.run();
        } finally {
            ACTIVE_PROFILE.set(active);
        }
    }

    static InferenceProfile startProfile(String mode) {
        InferenceProfile profile = new InferenceProfile(mode, profilingEnabled());
        ACTIVE_PROFILE.set(profile);
        return profile;
    }

    static void clearProfile() {
        ACTIVE_PROFILE.remove();
    }

    static void markFirstToken(InferenceProfile profile, long requestStartNanos) {
        if (profile != null && profile.firstTokenNanos <= 0L) {
            profile.firstTokenNanos = System.nanoTime() - requestStartNanos;
        }
    }

    static void recordModelLoadNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) {
            profile.modelLoadNanos += nanos;
        }
    }

    static void maybePrintProfileSummary(InferenceProfile profile, String backend) {
        if (profile != null && profilingEnabled()) {
            System.out.println("\n[PROFILE] " + profile.summary(backend));
            System.out.flush();
        }
    }

    static String backendLabel(Instance<?> metalBackend) {
        if (hasUsableMetalBackend(metalBackend) || isNativeMetalRuntimeActive()) {
            return "metal";
        }
        return "cpu";
    }

    static String metalDeviceLabel(Instance<?> metalBackend) {
        if (isNativeMetalRuntimeActive()) {
            try {
                return MetalBinding.getInstance().deviceName();
            } catch (Exception ignored) {
                // fall through
            }
        }
        if (metalBackend == null || !metalBackend.isResolvable()) {
            return null;
        }
        try {
            Object backend = metalBackend.get();
            return (String) backend.getClass().getMethod("deviceName").invoke(backend);
        } catch (Exception ignored) {
            return null;
        }
    }

    static void putGenerationBenchMetadata(Map<String, Object> metadata,
            int promptTokens,
            int completionTokens,
            long sessionAllocateNanos,
            long prefillNanos,
            long decodeNanos,
            long samplingNanos,
            long firstTokenNanos,
            int decodeSteps) {
        if (metadata == null) {
            return;
        }
        int outputTokens = Math.max(0, completionTokens);
        int measuredDecodeSteps = Math.max(0, decodeSteps);
        metadata.put("tokens.input", Math.max(0, promptTokens));
        metadata.put("tokens.output", outputTokens);
        metadata.put("tokens.decode", measuredDecodeSteps);
        if (sessionAllocateNanos > 0L) {
            metadata.put("bench.session_allocate_ms", nanosToMillis(sessionAllocateNanos));
        }
        if (firstTokenNanos > 0L) {
            metadata.put("bench.ttft_ms", nanosToMillis(firstTokenNanos));
            metadata.put("bench.cold_ttft_ms", nanosToMillis(firstTokenNanos));
        }
        if (promptTokens > 0 && prefillNanos > 0L) {
            double prefillSec = prefillNanos / 1_000_000_000.0;
            metadata.put("bench.prefill_tps", promptTokens / prefillSec);
            metadata.put("bench.prefill_ms_per_token", nanosToMillis(prefillNanos) / promptTokens);
            metadata.put("bench.prefill_ms", nanosToMillis(prefillNanos));
        }
        if (samplingNanos > 0L) {
            metadata.put("bench.sampling_ms", nanosToMillis(samplingNanos));
        }
        if (measuredDecodeSteps > 0 && decodeNanos > 0L) {
            double decodeSec = decodeNanos / 1_000_000_000.0;
            double decodeMsPerToken = nanosToMillis(decodeNanos) / measuredDecodeSteps;
            metadata.put("bench.decode_tps", measuredDecodeSteps / decodeSec);
            metadata.put("bench.decode_ms_per_token", decodeMsPerToken);
            metadata.put("bench.decode_ms", nanosToMillis(decodeNanos));
            metadata.put("bench.tpot_ms", decodeMsPerToken);
        }
        if (outputTokens > 0) {
            long generationNanos = prefillNanos + decodeNanos + samplingNanos;
            if (generationNanos > 0L) {
                metadata.put("bench.generation_tps", outputTokens / (generationNanos / 1_000_000_000.0));
            }
        }
    }

    private static boolean profilingEnabled() {
        return Boolean.getBoolean(PROFILE_PROPERTY);
    }

    private static boolean hasUsableMetalBackend(Instance<?> metalBackend) {
        if (metalBackend == null || !metalBackend.isResolvable()) {
            return false;
        }
        try {
            Object backend = metalBackend.get();
            String deviceName = (String) backend.getClass().getMethod("deviceName").invoke(backend);
            return deviceName != null && !deviceName.contains("CPU");
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean isNativeMetalRuntimeActive() {
        try {
            MetalBinding binding = MetalBinding.getInstance();
            return binding != null && binding.isRuntimeActive();
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }
}
