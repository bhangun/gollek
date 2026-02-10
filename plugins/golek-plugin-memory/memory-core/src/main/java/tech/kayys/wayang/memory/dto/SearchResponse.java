public record SearchResponse(
        boolean success,
        List<MemoryResult> results,
        int count) {
}