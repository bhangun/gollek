package tech.kayys.wayang.memory.context;

import java.time.Instant;

public class Fact {
    private final String subject;
    private final String predicate;
    private final String object;
    private final double confidence;
    private final Instant extractedAt;

    public Fact(String subject, String predicate, String object, double confidence, Instant extractedAt) {
        this.subject = subject;
        this.predicate = predicate;
        this.object = object;
        this.confidence = confidence;
        this.extractedAt = extractedAt;
    }

    public String getSubject() { return subject; }
    public String getPredicate() { return predicate; }
    public String getObject() { return object; }
    public double getConfidence() { return confidence; }
    public Instant getExtractedAt() { return extractedAt; }
}
