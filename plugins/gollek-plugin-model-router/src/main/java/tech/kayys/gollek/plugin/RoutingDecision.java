package tech.kayys.gollek.plugin;

import java.util.List;
import java.util.Map;

/**
 * Represents the outcome of a routing decision.
 */
public class RoutingDecision {

    private final String modelId;
    private final String providerId;
    private final String requestId;
    private final double score;
    private final List<String> candidates;
    private final Map<String, Object> metadata;
    private final long timestamp;

    public RoutingDecision(String modelId, String providerId, String requestId,
            double score, List<String> candidates, Map<String, Object> metadata) {
        this.modelId = modelId;
        this.providerId = providerId;
        this.requestId = requestId;
        this.score = score;
        this.candidates = candidates != null ? candidates : List.of();
        this.metadata = metadata != null ? metadata : Map.of();
        this.timestamp = System.currentTimeMillis();
    }

    public String getModelId() {
        return modelId;
    }

    public String getProviderId() {
        return providerId;
    }

    public String getRequestId() {
        return requestId;
    }

    public double getScore() {
        return score;
    }

    public List<String> getCandidates() {
        return candidates;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public boolean isValid() {
        return providerId != null && !providerId.trim().isEmpty();
    }

    public static RoutingDecisionBuilder builder() {
        return new RoutingDecisionBuilder();
    }

    public static class RoutingDecisionBuilder {
        private String modelId;
        private String providerId;
        private String requestId;
        private double score = 0.0;
        private List<String> candidates = List.of();
        private Map<String, Object> metadata = Map.of();

        public RoutingDecisionBuilder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public RoutingDecisionBuilder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public RoutingDecisionBuilder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public RoutingDecisionBuilder score(double score) {
            this.score = score;
            return this;
        }

        public RoutingDecisionBuilder candidates(List<String> candidates) {
            this.candidates = candidates;
            return this;
        }

        public RoutingDecisionBuilder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public RoutingDecision build() {
            return new RoutingDecision(modelId, providerId, requestId, score, candidates, metadata);
        }
    }
}