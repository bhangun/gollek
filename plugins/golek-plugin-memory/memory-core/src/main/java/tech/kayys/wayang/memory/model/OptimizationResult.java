package tech.kayys.wayang.memory.model;

import java.time.Instant;

public class OptimizationResult {
    private final String sessionId;
    private final int originalSize;
    private final int optimizedSize;
    private final long spaceSaved;
    private final Instant optimizedAt;

    public OptimizationResult(
            String sessionId,
            int originalSize,
            int optimizedSize,
            long spaceSaved,
            Instant optimizedAt) {
        this.sessionId = sessionId;
        this.originalSize = originalSize;
        this.optimizedSize = optimizedSize;
        this.spaceSaved = spaceSaved;
        this.optimizedAt = optimizedAt;
    }

    public String getSessionId() { return sessionId; }
    public int getOriginalSize() { return originalSize; }
    public int getOptimizedSize() { return optimizedSize; }
    public long getSpaceSaved() { return spaceSaved; }
    public Instant getOptimizedAt() { return optimizedAt; }
}