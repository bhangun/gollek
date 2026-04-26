package tech.kayys.gollek.tool.dto;

import io.quarkus.runtime.annotations.RegisterForReflection;

@RegisterForReflection
public enum InvocationStatus {
    SUCCESS,
    FAILURE,
    PENDING,
    TIMEOUT
}