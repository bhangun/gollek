package tech.kayys.wayang.tool.dto;

public record CreateAuthProfileRequest(
        String profileName,
        String authType,
        String location,
        String paramName,
        String scheme,
        String secretValue,
        String description) {
}