package tech.kayys.golek.engine.resource.dto;

import java.util.List;
import java.util.Map;
import jakarta.validation.constraints.NotBlank;

public record InferenceRequestDTO(
        @NotBlank String model,
        List<InferenceMessageDTO> messages,
        Map<String, Object> parameters,
        long timeout,
        int priority) {
}
