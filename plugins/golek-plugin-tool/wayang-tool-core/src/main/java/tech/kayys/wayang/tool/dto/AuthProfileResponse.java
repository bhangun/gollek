package tech.kayys.wayang.tool.dto;

public record AuthProfileResponse(
        String profileId,
        String profileName,
        String authType,
        boolean enabled) {
}