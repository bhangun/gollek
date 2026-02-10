package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;

public class ScoredMemory {
    private final ConversationMemory memory;
    private final double score;

    public ScoredMemory(ConversationMemory memory, double score) {
        this.memory = memory;
        this.score = score;
    }

    public ConversationMemory getMemory() { return memory; }
    public double getScore() { return score; }
}