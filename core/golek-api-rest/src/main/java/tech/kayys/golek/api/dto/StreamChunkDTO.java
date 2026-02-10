package tech.kayys.golek.api.dto;

public record StreamChunkDTO(int index, String content, boolean isFinal) {
}
