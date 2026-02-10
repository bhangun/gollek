package tech.kayys.wayang.mcp.dto;

public record CreateAuthProfileRequest(
        String profileName,
        String authType,
        String location,
        String paramName,
        String scheme,
        String secretValue,
        String description) {
}