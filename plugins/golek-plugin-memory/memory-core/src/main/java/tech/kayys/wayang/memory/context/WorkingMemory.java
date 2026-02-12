package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public class WorkingMemory {
    private final String currentTask;
    private final List<ConversationMemory> activeEpisodic;
    private final List<Fact> activeFacts;
    private final List<TaskPattern> activePatterns;
    private final Map<String, Double> attentionWeights;
    private final Instant timestamp;

    public WorkingMemory(String currentTask, List<ConversationMemory> activeEpisodic,
                        List<Fact> activeFacts, List<TaskPattern> activePatterns,
                        Map<String, Double> attentionWeights, Instant timestamp) {
        this.currentTask = currentTask;
        this.activeEpisodic = activeEpisodic;
        this.activeFacts = activeFacts;
        this.activePatterns = activePatterns;
        this.attentionWeights = attentionWeights;
        this.timestamp = timestamp;
    }

    public String getCurrentTask() { return currentTask; }
    public List<ConversationMemory> getActiveEpisodic() { return activeEpisodic; }
    public List<Fact> getActiveFacts() { return activeFacts; }
    public List<TaskPattern> getActivePatterns() { return activePatterns; }
    public Map<String, Double> getAttentionWeights() { return attentionWeights; }
    public Instant getTimestamp() { return timestamp; }
}
