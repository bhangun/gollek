package tech.kayys.gollek.sdk.mcp;

import java.util.List;

public record McpDoctorReport(
                List<McpDoctorEntry> entries,
                int passed,
                int failed,
                int total,
                String registryPath) {
}
