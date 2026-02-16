package tech.kayys.wayang.tool.dto;

import java.util.Map;

/**
 * Tool execution request
 */
public record ToolExecutionRequest(
                String requestId,
                String userId,
                String toolId,
                Map<String, Object> arguments,
                String workflowRunId,
                String agentId,
                Map<String, Object> context) {
}
