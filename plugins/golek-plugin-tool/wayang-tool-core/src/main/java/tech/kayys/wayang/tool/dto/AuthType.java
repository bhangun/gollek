package tech.kayys.wayang.tool.dto;

public enum AuthType {
    API_KEY,
    BEARER_TOKEN,
    BASIC_AUTH,
    OAUTH2_CLIENT_CREDENTIALS,
    OAUTH2_AUTHORIZATION_CODE,
    CUSTOM_HEADER
}