package tech.kayys.wayang.memory.dto;

import tech.kayys.wayang.memory.model.MemoryType;
import java.util.Map;

public record StoreRequest(

        String namespace,
        String content,
        MemoryType type,
        Double importance,
        Map<String, Object> metadata) {
}
