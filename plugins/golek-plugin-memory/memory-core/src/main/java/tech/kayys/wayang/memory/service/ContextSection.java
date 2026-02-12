package tech.kayys.wayang.memory.service;


/**
 * Context section
 */
public class ContextSection {

    private final String type;
    private final String content;
    private final int tokenCount;
    private final double relevanceScore;

    public ContextSection(String type, String content, int tokenCount, double relevanceScore) {
        this.type = type;
        this.content = content;
        this.tokenCount = tokenCount;
        this.relevanceScore = relevanceScore;
    }

    public String getType() {
        return type;
    }

    public String getContent() {
        return content;
    }

    public int getTokenCount() {
        return tokenCount;
    }

    public double getRelevanceScore() {
        return relevanceScore;
    }
}