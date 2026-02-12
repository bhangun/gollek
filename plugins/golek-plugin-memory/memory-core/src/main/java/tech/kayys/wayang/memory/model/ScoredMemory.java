package tech.kayys.wayang.memory.model;

import java.util.Map;

/**
 * A memory with an associated score from search or ranking operations
 */
public class ScoredMemory {
    private final Memory memory;
    private final double score;
    private final Map<String, Object> scoreBreakdown;

    public ScoredMemory(Memory memory, double score) {
        this(memory, score, Map.of());
    }

    public ScoredMemory(Memory memory, double score, Map<String, Object> scoreBreakdown) {
        this.memory = memory;
        this.score = score;
        this.scoreBreakdown = scoreBreakdown != null ? scoreBreakdown : Map.of();
    }

    public Memory getMemory() {
        return memory;
    }

    public double getScore() {
        return score;
    }

    public Map<String, Object> getScoreBreakdown() {
        return scoreBreakdown;
    }
}
