package tech.kayys.wayang.memory.dto;

public record ExampleResponse(

        boolean success,
        String message,
        int examplesCount) {
}