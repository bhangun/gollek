package tech.kayys.golek.memory;

/**
 * Strategy for ranking memory retrieval results.
 */
public enum MemoryRankingStrategy {

    /**
     * Semantic relevance (cosine similarity)
     */
    RELEVANCE,

    /**
     * Most recent first
     */
    RECENCY,

    /**
     * Most important first
     */
    IMPORTANCE,

    /**
     * Most frequently accessed
     */
    FREQUENCY,

    /**
     * Hybrid: relevance + recency + importance
     */
    HYBRID;

    /**
     * Calculate score for memory entry
     */
    public double calculateScore(MemoryEntry entry, MemoryQuery query) {
        return switch (this) {
            case RELEVANCE -> calculateRelevanceScore(entry, query);
            case RECENCY -> entry.getRecencyScore();
            case IMPORTANCE -> entry.getImportance();
            case FREQUENCY -> entry.getFrequencyScore();
            case HYBRID -> calculateHybridScore(entry, query);
        };
    }

    private double calculateRelevanceScore(MemoryEntry entry, MemoryQuery query) {
        if (query.getQueryEmbedding().isEmpty() || !entry.hasEmbedding()) {
            return 0.5; // Default middle score
        }
        return cosineSimilarity(
                entry.getEmbedding(),
                query.getQueryEmbedding().get());
    }

    private double calculateHybridScore(MemoryEntry entry, MemoryQuery query) {
        double relevance = calculateRelevanceScore(entry, query);
        double recency = entry.getRecencyScore();
        double importance = entry.getImportance();

        // Weighted combination
        return (relevance * 0.5) + (recency * 0.3) + (importance * 0.2);
    }

    private double cosineSimilarity(List<Double> v1, List<Double> v2) {
        if (v1.size() != v2.size()) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < v1.size(); i++) {
            dotProduct += v1.get(i) * v2.get(i);
            norm1 += v1.get(i) * v1.get(i);
            norm2 += v2.get(i) * v2.get(i);
        }

        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
}