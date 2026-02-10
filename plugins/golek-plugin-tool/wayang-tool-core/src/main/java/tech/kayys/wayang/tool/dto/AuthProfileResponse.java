package tech.kayys.wayang.mcp.dto;

public record AuthProfileResponse(
        String profileId,
        String profileName,
        String authType,
        boolean enabled) {
}