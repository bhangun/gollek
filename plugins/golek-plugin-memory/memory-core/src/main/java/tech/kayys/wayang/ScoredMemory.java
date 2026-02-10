package tech.kayys.gamelan.executor.memory;

import java.util.Map;

/**
 * A memory with an associated score from search or ranking operations
 */
public class ScoredMemory {
    private final Memory memory;
    private final double score;
    private final Map<String, Object> scoreBreakdown;

    public ScoredMemory(Memory memory, double score) {
        this(memory, score, null);
    }

    public ScoredMemory(Memory memory, double score, Map<String, Object> scoreBreakdown) {
        this.memory = memory;
        this.score = score;
        this.scoreBreakdown = scoreBreakdown;
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