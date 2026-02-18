package tech.kayys.gollek.api.dto;

public record StreamChunkDTO(int index, String content, boolean isFinal) {
}
