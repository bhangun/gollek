package tech.kayys.golek.engine.resource.dto;

public record StreamChunkDTO(int index, String content, boolean isFinal) {
}
