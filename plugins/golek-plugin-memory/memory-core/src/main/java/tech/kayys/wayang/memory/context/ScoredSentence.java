package tech.kayys.wayang.memory.context;

public class ScoredSentence {
    private final String sentence;
    private final double score;
    private final String sourceId;

    public ScoredSentence(String sentence, double score, String sourceId) {
        this.sentence = sentence;
        this.score = score;
        this.sourceId = sourceId;
    }

    public String getSentence() { return sentence; }
    public double getScore() { return score; }
    public String getSourceId() { return sourceId; }
}