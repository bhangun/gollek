package tech.kayys.gollek.waitscheduler.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Heuristic fallback for ForeLen when the native predictor is unavailable.
 *
 * <p>Formula: {@code predicted = clamp(promptTokens / 2, 64, 2048)}
 * This approximates the typical ratio of output length to prompt length
 * observed across common LLM workloads.
 */
public final class ForelenHeuristic {

    private ForelenHeuristic() {}

    public static int predict(int promptTokens) {
        return Math.max(64, Math.min(2048, promptTokens / 2));
    }

    public static int predictBatch(MemorySegment out, MemorySegment in, int n) {
        for (int i = 0; i < n; i++) {
            int p = in.getAtIndex(ValueLayout.JAVA_INT, i);
            out.setAtIndex(ValueLayout.JAVA_INT, i, predict(p));
        }
        return 0;
    }
}
