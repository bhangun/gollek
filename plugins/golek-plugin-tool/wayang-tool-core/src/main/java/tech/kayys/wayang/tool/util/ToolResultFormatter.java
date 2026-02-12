package tech.kayys.wayang.tool.util;

import tech.kayys.wayang.tool.dto.ToolExecutionResult;
import tech.kayys.wayang.tool.spi.Tool;

import java.util.List;
import java.util.Map;

/**
 * Utility for formatting tool execution results for injection back into the conversation.
 */
public class ToolResultFormatter {

    /**
     * Formats tool execution results into a structured format suitable for LLM consumption.
     *
     * @param toolResults the results from tool executions
     * @return formatted results as a list of maps containing tool call IDs and outputs
     */
    public List<Map<String, Object>> formatResults(List<ToolExecutionResult> toolResults) {
        return toolResults.stream()
                .map(this::formatSingleResult)
                .toList();
    }

    /**
     * Formats a single tool execution result.
     *
     * @param result the tool execution result
     * @return formatted result as a map
     */
    private Map<String, Object> formatSingleResult(ToolExecutionResult result) {
        return Map.of(
                "tool_call_id", result.toolId(),
                "name", result.toolId(), // Assuming toolId is used for name if name is missing in record
                "content", result.output()
        );
    }

    /**
     * Creates a structured message containing tool results for injection into conversation history.
     *
     * @param toolResults the results from tool executions
     * @return a structured message containing the tool results
     */
    public Map<String, Object> createToolResultMessage(List<ToolExecutionResult> toolResults) {
        return Map.of(
                "role", "tool",
                "content", formatResults(toolResults)
        );
    }
}