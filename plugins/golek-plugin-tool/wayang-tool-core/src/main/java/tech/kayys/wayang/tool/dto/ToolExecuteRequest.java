package tech.kayys.wayang.tool.dto;

import java.util.Map;

public record ToolExecuteRequest(
        Map<String, Object> arguments,
        Map<String, Object> context) {
}