public record ContextResponse(
        boolean success,
        String prompt,
        int totalTokens,
        double utilization,
        int sectionCount) {
}