package tech.kayys.golek.memory;

import java.time.Instant;
import java.util.*;

/**
 * Query specification for memory retrieval.
 * Supports filtering, ranking, and pagination.
 */
public final class MemoryQuery {

    private final Set<MemoryType> types;
    private final String textQuery;
    private final List<Double> queryEmbedding;
    private final Set<String> tags;
    private final Map<String, Object> metadataFilters;

    private final String tenantId;
    private final String userId;
    private final String sessionId;

    private final Instant afterTimestamp;
    private final Instant beforeTimestamp;

    private final Double minImportance;
    private final int limit;
    private final int offset;

    private final MemoryRankingStrategy rankingStrategy;

    private MemoryQuery(Builder builder) {
        this.types = Collections.unmodifiableSet(new HashSet<>(builder.types));
        this.textQuery = builder.textQuery;
        this.queryEmbedding = builder.queryEmbedding != null
                ? Collections.unmodifiableList(new ArrayList<>(builder.queryEmbedding))
                : null;
        this.tags = Collections.unmodifiableSet(new HashSet<>(builder.tags));
        this.metadataFilters = Collections.unmodifiableMap(new HashMap<>(builder.metadataFilters));
        this.tenantId = builder.tenantId;
        this.userId = builder.userId;
        this.sessionId = builder.sessionId;
        this.afterTimestamp = builder.afterTimestamp;
        this.beforeTimestamp = builder.beforeTimestamp;
        this.minImportance = builder.minImportance;
        this.limit = builder.limit;
        this.offset = builder.offset;
        this.rankingStrategy = builder.rankingStrategy;
    }

    // Getters
    public Set<MemoryType> getTypes() {
        return types;
    }

    public Optional<String> getTextQuery() {
        return Optional.ofNullable(textQuery);
    }

    public Optional<List<Double>> getQueryEmbedding() {
        return Optional.ofNullable(queryEmbedding);
    }

    public Set<String> getTags() {
        return tags;
    }

    public Map<String, Object> getMetadataFilters() {
        return metadataFilters;
    }

    public Optional<String> getTenantId() {
        return Optional.ofNullable(tenantId);
    }

    public Optional<String> getUserId() {
        return Optional.ofNullable(userId);
    }

    public Optional<String> getSessionId() {
        return Optional.ofNullable(sessionId);
    }

    public Optional<Instant> getAfterTimestamp() {
        return Optional.ofNullable(afterTimestamp);
    }

    public Optional<Instant> getBeforeTimestamp() {
        return Optional.ofNullable(beforeTimestamp);
    }

    public Optional<Double> getMinImportance() {
        return Optional.ofNullable(minImportance);
    }

    public int getLimit() {
        return limit;
    }

    public int getOffset() {
        return offset;
    }

    public MemoryRankingStrategy getRankingStrategy() {
        return rankingStrategy;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final Set<MemoryType> types = new HashSet<>();
        private String textQuery;
        private List<Double> queryEmbedding;
        private final Set<String> tags = new HashSet<>();
        private final Map<String, Object> metadataFilters = new HashMap<>();
        private String tenantId;
        private String userId;
        private String sessionId;
        private Instant afterTimestamp;
        private Instant beforeTimestamp;
        private Double minImportance;
        private int limit = 10;
        private int offset = 0;
        private MemoryRankingStrategy rankingStrategy = MemoryRankingStrategy.RELEVANCE;

        public Builder type(MemoryType type) {
            this.types.add(type);
            return this;
        }

        public Builder types(Set<MemoryType> types) {
            this.types.addAll(types);
            return this;
        }

        public Builder textQuery(String textQuery) {
            this.textQuery = textQuery;
            return this;
        }

        public Builder queryEmbedding(List<Double> queryEmbedding) {
            this.queryEmbedding = queryEmbedding;
            return this;
        }

        public Builder tag(String tag) {
            this.tags.add(tag);
            return this;
        }

        public Builder tags(Set<String> tags) {
            this.tags.addAll(tags);
            return this;
        }

        public Builder metadataFilter(String key, Object value) {
            this.metadataFilters.put(key, value);
            return this;
        }

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder afterTimestamp(Instant afterTimestamp) {
            this.afterTimestamp = afterTimestamp;
            return this;
        }

        public Builder beforeTimestamp(Instant beforeTimestamp) {
            this.beforeTimestamp = beforeTimestamp;
            return this;
        }

        public Builder minImportance(double minImportance) {
            this.minImportance = minImportance;
            return this;
        }

        public Builder limit(int limit) {
            this.limit = limit;
            return this;
        }

        public Builder offset(int offset) {
            this.offset = offset;
            return this;
        }

        public Builder rankingStrategy(MemoryRankingStrategy strategy) {
            this.rankingStrategy = strategy;
            return this;
        }

        public MemoryQuery build() {
            return new MemoryQuery(this);
        }
    }

    @Override
    public String toString() {
        return "MemoryQuery{" +
                "types=" + types +
                ", limit=" + limit +
                ", rankingStrategy=" + rankingStrategy +
                '}';
    }
}