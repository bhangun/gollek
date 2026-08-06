package tech.kayys.gollek.sdk.mcp;

import java.util.List;

public record McpTestReport(
                List<McpTestEntry> entries,
                int passed,
                int failed,
                int total) {
}
