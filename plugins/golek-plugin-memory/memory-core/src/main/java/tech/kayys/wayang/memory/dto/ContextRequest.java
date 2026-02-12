package tech.kayys.wayang.memory.dto;

public record ContextRequest(

        String namespace,
        String query,
        Integer maxMemories,
        String systemPrompt,
        String taskInstructions,
        Boolean includeMetadata) {
}
