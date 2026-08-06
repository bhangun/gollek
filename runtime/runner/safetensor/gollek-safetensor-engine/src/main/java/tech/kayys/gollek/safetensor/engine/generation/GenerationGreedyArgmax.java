/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.BitSet;

final class GenerationGreedyArgmax {
    private static final String DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY = "gollek.safetensor.disable_native_greedy_argmax";
    private static final boolean DISABLE_NATIVE_GREEDY_ARGMAX_ENABLED = Boolean
            .getBoolean(DISABLE_NATIVE_GREEDY_ARGMAX_PROPERTY);
    private static final int MAX_NATIVE_REJECTED_CANDIDATES = 8;
    private static final int NATIVE_ARGMAX_UNAVAILABLE = Integer.MIN_VALUE;
    private static final String NATIVE_ARGMAX_PATH = "native_argmax_f32";
    private static final String JAVA_MEMORY_SEGMENT_PATH = "java_memory_segment";
    private static final String JAVA_MEMORY_SEGMENT_MASK_PATH = "java_memory_segment_mask";
    private static final ThreadLocal<int[]> NATIVE_REJECTION_BUFFER = ThreadLocal
            .withInitial(() -> new int[MAX_NATIVE_REJECTED_CANDIDATES]);
    private static volatile Boolean nativeArgmaxDisabledForTest;

    private GenerationGreedyArgmax() {
    }

    static int select(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        long startedNanos = DirectInferenceProfiler.startGreedyArgmaxTiming();
        int nativeBest = tryNative(seg, vocab, rejectedCandidates, rejectedCount);
        if (nativeBest != NATIVE_ARGMAX_UNAVAILABLE) {
            recordProfile(startedNanos, NATIVE_ARGMAX_PATH);
            return nativeBest;
        }
        int javaBest = selectJava(seg, vocab, rejectedCandidates, rejectedCount);
        recordProfile(startedNanos, JAVA_MEMORY_SEGMENT_PATH);
        return javaBest;
    }

    static int selectJava(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        return selectJava(seg, vocab, null, rejectedCandidates, rejectedCount);
    }

    static int selectWithMask(
            MemorySegment seg,
            long vocab,
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount) {
        long startedNanos = DirectInferenceProfiler.startGreedyArgmaxTiming();
        int nativeBest = tryNativeWithMask(seg, vocab, rejectedMask, rejectedCandidates, rejectedCount);
        if (nativeBest != NATIVE_ARGMAX_UNAVAILABLE) {
            recordProfile(startedNanos, NATIVE_ARGMAX_PATH);
            return nativeBest;
        }
        int best = selectJava(seg, vocab, rejectedMask, rejectedCandidates, rejectedCount);
        recordProfile(startedNanos, JAVA_MEMORY_SEGMENT_MASK_PATH);
        return best;
    }

    static int selectJava(
            MemorySegment seg,
            long vocab,
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount) {
        int limit = javaVocabLimit(vocab);
        if (limit < 0) {
            return -1;
        }
        if (rejectedMask == null && rejectedCount <= 0) {
            return selectJavaUnmasked(seg, limit);
        }
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < limit; i++) {
            if (isRejected(rejectedMask, rejectedCandidates, rejectedCount, i)) {
                continue;
            }
            // Logit softcap is monotonic, so it cannot change greedy argmax ordering.
            float value = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    static int selectJava(float[] logits) {
        return selectJava(logits, logits.length, null, 0);
    }

    static int selectJava(
            float[] logits,
            int vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        if (rejectedCount <= 0) {
            return selectJavaUnmasked(logits, vocab);
        }
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        int limit = Math.min(vocab, logits.length);
        for (int i = 0; i < limit; i++) {
            if (rejectedCount > 0 && containsRejectedCandidate(rejectedCandidates, rejectedCount, i)) {
                continue;
            }
            float value = logits[i];
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    private static int selectJavaUnmasked(MemorySegment seg, int vocab) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            // Logit softcap is monotonic, so it cannot change greedy argmax ordering.
            float value = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    private static int selectJavaUnmasked(float[] logits, int vocab) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        int limit = Math.min(vocab, logits.length);
        for (int i = 0; i < limit; i++) {
            float value = logits[i];
            if (Float.isNaN(value)) {
                continue;
            }
            if (value > bestVal) {
                bestVal = value;
                best = i;
            }
        }
        return best;
    }

    private static int tryNative(
            MemorySegment seg,
            long vocab,
            int[] rejectedCandidates,
            int rejectedCount) {
        if (nativeArgmaxDisabled() || vocab > Integer.MAX_VALUE) {
            return NATIVE_ARGMAX_UNAVAILABLE;
        }
        try {
            MetalBinding binding = MetalBinding.getInstance();
            if (binding == null || !binding.supportsArgmaxF32()) {
                return NATIVE_ARGMAX_UNAVAILABLE;
            }
            return binding.argmaxF32(
                    seg,
                    (int) vocab,
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 0),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 1),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 2),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 3),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 4),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 5),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 6),
                    rejectedCandidateAt(rejectedCandidates, rejectedCount, 7));
        } catch (Throwable ignored) {
            return NATIVE_ARGMAX_UNAVAILABLE;
        }
    }

    private static int tryNativeWithMask(
            MemorySegment seg,
            long vocab,
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount) {
        int[] nativeRejections = NATIVE_REJECTION_BUFFER.get();
        int nativeRejectedCount = compactNativeRejections(
                vocab, rejectedMask, rejectedCandidates, rejectedCount, nativeRejections);
        if (nativeRejectedCount < 0) {
            return NATIVE_ARGMAX_UNAVAILABLE;
        }
        return tryNative(seg, vocab, nativeRejections, nativeRejectedCount);
    }

    static int compactNativeRejectionsForTest(
            long vocab,
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount,
            int[] compact) {
        return compactNativeRejections(vocab, rejectedMask, rejectedCandidates, rejectedCount, compact);
    }

    private static int compactNativeRejections(
            long vocab,
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount,
            int[] compact) {
        if (vocab > Integer.MAX_VALUE) {
            return -1;
        }
        int count = 0;
        int vocabLimit = (int) vocab;
        if (rejectedCandidates != null) {
            for (int i = 0; i < rejectedCount; i++) {
                int tokenId = rejectedCandidates[i];
                if (tokenId >= 0 && tokenId < vocabLimit) {
                    count = appendNativeRejection(compact, count, tokenId);
                    if (count < 0) {
                        return -1;
                    }
                }
            }
        }
        if (rejectedMask != null) {
            for (int tokenId = rejectedMask.nextSetBit(0); tokenId >= 0
                    && tokenId < vocabLimit; tokenId = rejectedMask.nextSetBit(tokenId + 1)) {
                count = appendNativeRejection(compact, count, tokenId);
                if (count < 0) {
                    return -1;
                }
            }
        }
        return count;
    }

    private static int appendNativeRejection(int[] compact, int count, int tokenId) {
        for (int i = 0; i < count; i++) {
            if (compact[i] == tokenId) {
                return count;
            }
        }
        if (count >= compact.length) {
            return -1;
        }
        compact[count] = tokenId;
        return count + 1;
    }

    private static void recordProfile(long startedNanos, String path) {
        DirectInferenceProfiler.recordGreedyArgmaxTiming(startedNanos, path);
    }

    private static boolean nativeArgmaxDisabled() {
        Boolean override = nativeArgmaxDisabledForTest;
        return override == null ? DISABLE_NATIVE_GREEDY_ARGMAX_ENABLED : override;
    }

    static void setNativeArgmaxDisabledForTest(Boolean disabled) {
        nativeArgmaxDisabledForTest = disabled;
    }

    private static int javaVocabLimit(long vocab) {
        if (vocab <= 0 || vocab > Integer.MAX_VALUE) {
            return -1;
        }
        return (int) vocab;
    }

    private static int rejectedCandidateAt(int[] rejectedCandidates, int rejectedCount, int index) {
        if (rejectedCandidates == null || index >= rejectedCount) {
            return -1;
        }
        return rejectedCandidates[index];
    }

    private static boolean containsRejectedCandidate(int[] rejectedCandidates, int rejectedCount, int tokenId) {
        if (rejectedCandidates == null) {
            return false;
        }
        for (int i = 0; i < rejectedCount; i++) {
            if (rejectedCandidates[i] == tokenId) {
                return true;
            }
        }
        return false;
    }

    private static boolean isRejected(
            BitSet rejectedMask,
            int[] rejectedCandidates,
            int rejectedCount,
            int tokenId) {
        return rejectedMask != null && rejectedMask.get(tokenId)
                || rejectedCount > 0 && containsRejectedCandidate(rejectedCandidates, rejectedCount, tokenId);
    }
}
