package tech.kayys.wayang.mcp.dto;

public enum AuthType {
    API_KEY,
    BEARER_TOKEN,
    BASIC_AUTH,
    OAUTH2_CLIENT_CREDENTIALS,
    OAUTH2_AUTHORIZATION_CODE,
    CUSTOM_HEADER
}