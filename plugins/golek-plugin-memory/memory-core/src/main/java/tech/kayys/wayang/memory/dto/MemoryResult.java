package tech.kayys.wayang.memory.dto;

public record MemoryResult(

        String id,
        String content,
        double score,
        String type,
        double importance) {
}
