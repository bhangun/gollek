package tech.kayys.golek.api.dto;

public record ErrorResponseDTO(String message, String errorType) {
    public static ErrorResponseDTO from(Throwable t) {
        return new ErrorResponseDTO(t.getMessage(), t.getClass().getSimpleName());
    }
}
