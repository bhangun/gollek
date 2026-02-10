package tech.kayys.wayang.memory.context;

import java.time.Instant;

public class MemoryStrength {
    private final String memoryId;
    private final double strength;
    private final double decayRate;
    private final int accessCount;
    private final Instant nextReviewTime;

    public MemoryStrength(String memoryId, double strength, double decayRate,
                         int accessCount, Instant nextReviewTime) {
        this.memoryId = memoryId;
        this.strength = strength;
        this.decayRate = decayRate;
        this.accessCount = accessCount;
        this.nextReviewTime = nextReviewTime;
    }

    public String getMemoryId() { return memoryId; }
    public double getStrength() { return strength; }
    public double getDecayRate() { return decayRate; }
    public int getAccessCount() { return accessCount; }
    public Instant getNextReviewTime() { return nextReviewTime; }
}