package tech.kayys.wayang.mcp.dto;

import java.util.Map;

public record ToolExecuteRequest(
        Map<String, Object> arguments,
        Map<String, Object> context) {
}