public record StoreRequest(
        String namespace,
        String content,
        MemoryType type,
        Double importance,
        Map<String, Object> metadata) {
}
