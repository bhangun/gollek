package tech.kayys.wayang.memory.context;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public class SemanticMemory {
    private final Map<String, List<String>> knowledgeGraph;
    private final List<Fact> facts;
    private final Map<String, Double> conceptStrength;
    private final Instant timestamp;

    public SemanticMemory(Map<String, List<String>> knowledgeGraph, List<Fact> facts,
                         Map<String, Double> conceptStrength, Instant timestamp) {
        this.knowledgeGraph = knowledgeGraph;
        this.facts = facts;
        this.conceptStrength = conceptStrength;
        this.timestamp = timestamp;
    }

    public Map<String, List<String>> getKnowledgeGraph() { return knowledgeGraph; }
    public List<Fact> getFacts() { return facts; }
    public Map<String, Double> getConceptStrength() { return conceptStrength; }
    public Instant getTimestamp() { return timestamp; }
}
