package tech.kayys.wayang.memory.spi;

import java.time.Instant;
import java.util.Map;

/**
 * Represents a unit of information in the agent's memory.
 */
public record MemoryEntry(
        String id,
        String content,
        Instant timestamp,
        Map<String, Object> metadata) {
}
