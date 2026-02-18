package tech.kayys.gollek.sdk.core.mcp;

import java.util.List;

public record McpDoctorReport(
        List<McpDoctorEntry> entries,
        int passed,
        int failed,
        int total,
        String registryPath) {
}

