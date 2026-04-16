package tech.kayys.gollek.cache;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Serializes and deserializes {@link CachedKVEntry} to/from a compact JSON string.
 *
 * <p>Used by the Redis and Disk stores. The in-process store keeps Java objects
 * directly in the Caffeine heap and does not use this serializer.
 *
 * <p>Format (all fields serialized):
 * <pre>
 * {
 *   "modelId": "Qwen2.5-0.5B",
 *   "tokenHash": 3456789012345678,
 *   "prefixLength": 128,
 *   "scope": "global",
 *   "blockIds": [4, 7, 19],
 *   "tokenCount": 128,
 *   "createdAt": "2026-03-15T10:30:00Z",
 *   "lastAccessedAt": "2026-03-15T10:31:00Z",
 *   "hitCount": 3
 * }
 * </pre>
 *
 * <p>Uses Jackson with Java 8 Time module. Quarkus provides the
 * application-scoped {@link ObjectMapper} automatically.
 */
@ApplicationScoped
public class CacheEntrySerializer {

    private static final Logger LOG = Logger.getLogger(CacheEntrySerializer.class);

    private final ObjectMapper mapper;

    @Inject
    public CacheEntrySerializer(ObjectMapper mapper) {
        this.mapper = mapper.copy()
                .registerModule(new JavaTimeModule());
    }

    public String serialize(CachedKVEntry entry) {
        try {
            Map<String, Object> map = Map.of(
                    "modelId",        entry.key().modelId(),
                    "tokenHash",      entry.key().tokenHash(),
                    "prefixLength",   entry.key().prefixLength(),
                    "scope",          entry.scope(),
                    "blockIds",       entry.blockIds(),
                    "tokenCount",     entry.tokenCount(),
                    "createdAt",      entry.createdAt().toString(),
                    "lastAccessedAt", entry.lastAccessedAt().toString(),
                    "hitCount",       entry.hitCount()
            );
            return mapper.writeValueAsString(map);
        } catch (Exception e) {
            LOG.errorf(e, "Failed to serialize CachedKVEntry");
            throw new RuntimeException("Serialization failed", e);
        }
    }

    @SuppressWarnings("unchecked")
    public CachedKVEntry deserialize(String json) {
        try {
            Map<String, Object> map = mapper.readValue(json, Map.class);

            PrefixHash key = new PrefixHash(
                    (String) map.get("modelId"),
                    ((Number) map.get("tokenHash")).longValue(),
                    ((Number) map.get("prefixLength")).intValue(),
                    (String) map.get("scope")
            );

            List<Integer> blockIds = ((List<?>) map.get("blockIds"))
                    .stream().map(o -> ((Number) o).intValue()).toList();

            return new CachedKVEntry(
                    key,
                    blockIds,
                    ((Number) map.get("tokenCount")).intValue(),
                    Instant.parse((String) map.get("createdAt")),
                    Instant.parse((String) map.get("lastAccessedAt")),
                    ((Number) map.get("hitCount")).longValue(),
                    (String) map.get("scope")
            );
        } catch (Exception e) {
            LOG.errorf(e, "Failed to deserialize CachedKVEntry from: %s",
                    json.length() > 200 ? json.substring(0, 200) + "..." : json);
            throw new RuntimeException("Deserialization failed", e);
        }
    }
}
