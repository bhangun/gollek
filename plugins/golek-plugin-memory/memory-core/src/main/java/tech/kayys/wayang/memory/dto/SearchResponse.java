package tech.kayys.wayang.memory.dto;

import java.util.List;

public record SearchResponse(

        boolean success,
        List<MemoryResult> results,
        int count) {
}