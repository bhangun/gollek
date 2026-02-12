package tech.kayys.wayang.memory.dto;

public record SearchRequest(

        String namespace,
        String query,
        Integer limit,
        Double minSimilarity) {
}
