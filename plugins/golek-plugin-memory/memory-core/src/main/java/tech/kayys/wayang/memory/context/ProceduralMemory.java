package tech.kayys.wayang.memory.context;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public class ProceduralMemory {
    private final String userId;
    private final List<TaskPattern> patterns;
    private final Map<String, List<String>> commonSequences;
    private final Map<String, Double> skillProficiency;
    private final Instant timestamp;

    public ProceduralMemory(String userId, List<TaskPattern> patterns,
                           Map<String, List<String>> commonSequences,
                           Map<String, Double> skillProficiency, Instant timestamp) {
        this.userId = userId;
        this.patterns = patterns;
        this.commonSequences = commonSequences;
        this.skillProficiency = skillProficiency;
        this.timestamp = timestamp;
    }

    public String getUserId() { return userId; }
    public List<TaskPattern> getPatterns() { return patterns; }
    public Map<String, List<String>> getCommonSequences() { return commonSequences; }
    public Map<String, Double> getSkillProficiency() { return skillProficiency; }
    public Instant getTimestamp() { return timestamp; }
}
