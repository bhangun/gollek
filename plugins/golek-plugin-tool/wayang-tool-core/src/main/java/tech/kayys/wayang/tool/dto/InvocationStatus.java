package tech.kayys.wayang.tool.dto;

public enum InvocationStatus {
    SUCCESS,
    FAILURE,
    TIMEOUT,
    RATE_LIMITED,
    UNAUTHORIZED,
    VALIDATION_ERROR,
    AUTH_ERROR,
    SECURITY_VIOLATION
}