package tech.kayys.wayang.memory.service;


import java.time.Instant;

/**
 * Statistics about memory storage and usage
 */
public class MemoryStatistics {
    private final String namespace;
    private final long totalMemories;
    private final long episodicCount;
    private final long semanticCount;
    private final long proceduralCount;
    private final long workingCount;
    private final double avgImportance;
    private final Instant oldestMemory;
    private final Instant newestMemory;

    public MemoryStatistics(
            String namespace,
            long totalMemories,
            long episodicCount,
            long semanticCount,
            long proceduralCount,
            long workingCount,
            double avgImportance,
            Instant oldestMemory,
            Instant newestMemory) {
        this.namespace = namespace;
        this.totalMemories = totalMemories;
        this.episodicCount = episodicCount;
        this.semanticCount = semanticCount;
        this.proceduralCount = proceduralCount;
        this.workingCount = workingCount;
        this.avgImportance = avgImportance;
        this.oldestMemory = oldestMemory;
        this.newestMemory = newestMemory;
    }

    public String getNamespace() {
        return namespace;
    }

    public long getTotalMemories() {
        return totalMemories;
    }

    public long getEpisodicCount() {
        return episodicCount;
    }

    public long getSemanticCount() {
        return semanticCount;
    }

    public long getProceduralCount() {
        return proceduralCount;
    }

    public long getWorkingCount() {
        return workingCount;
    }

    public double getAvgImportance() {
        return avgImportance;
    }

    public Instant getOldestMemory() {
        return oldestMemory;
    }

    public Instant getNewestMemory() {
        return newestMemory;
    }
}